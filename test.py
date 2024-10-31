import onnxruntime as ort
from utils import safe_load_image
from tokenizer import BertTokenizer
import numpy as np
from PIL import Image

def generate_text_token_mask(input_ids, special_token_ids):
    seq_len = input_ids.shape[1]
    text_token_mask = np.ones((1, seq_len, seq_len), dtype=bool)
    for i in range(seq_len):
        for j in range(seq_len):
            if input_ids[0, j] in special_token_ids:
                text_token_mask[0, i, j] = False
    # Allow self-attention for special tokens
    for idx in range(seq_len):
        if input_ids[0, idx] in special_token_ids:
            text_token_mask[0, idx, idx] = True
    return text_token_mask

def resize_image(image: Image.Image, target_size=800, max_size=1333):
    w, h = image.size
    min_original_size = float(min((w, h)))
    max_original_size = float(max((w, h)))
    if max_original_size / min_original_size * target_size > max_size:
        target_size = int(round(max_size * min_original_size / max_original_size))

    scale = target_size / min_original_size
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return image.resize((new_w, new_h), Image.BILINEAR)

def preprocess_image(image_path):
    image_pil = safe_load_image(image_path, return_numpy=False)
    image_resized = resize_image(image_pil)
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_np = (image_np - mean) / std
    image_np = np.transpose(image_np, (2, 0, 1))
    img = image_np[np.newaxis, :, :, :]
    return img



def get_phrases_from_posmap(posmap, tokenized, tokenizer, left_idx=0, right_idx=255):
    """
    Extracts phrases from the position map.

    Args:
        posmap (np.ndarray): Boolean array indicating positive positions. Shape: (seq_len,)
        tokenized (dict): Tokenized input containing 'input_ids' (list of token IDs).
        tokenizer (BertTokenizer): Your custom tokenizer.
        left_idx (int): Index to start considering tokens (usually skip special tokens).
        right_idx (int): Index to stop considering tokens.

    Returns:
        str: Decoded phrase.
    """
    # Ensure posmap is 1-dimensional
    if posmap.ndim != 1:
        raise NotImplementedError("posmap must be 1-dimensional")

    # Adjust posmap to ignore special tokens
    posmap[:left_idx + 1] = False  # Exclude tokens before and including left_idx
    posmap[right_idx:] = False     # Exclude tokens from right_idx onwards

    # Get indices where posmap is True
    non_zero_idx = np.nonzero(posmap)[0]

    # If no positive positions, return an empty string
    if len(non_zero_idx) == 0:
        return ""

    # Extract token IDs
    input_ids = tokenized["input_ids"][0]  # Assuming batch size of 1
    token_ids = [input_ids[i] for i in non_zero_idx]

    # Decode tokens to get the phrase
    phrase = tokenizer.decode(token_ids)

    return phrase


if __name__ == "__main__":

    import IPython

    # load the model
    model_path = "./weights/grounded.onnx"

    session = ort.InferenceSession(
                model_path,
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if ort.get_device() == "GPU"
                    else ["CPUExecutionProvider"]
                ),
            )

    # needs inference, but i need img, input_ids, attention_mask, token_type_ids, text_token_mask
    tokenizer = BertTokenizer(vocab_file="./weights/vocab.txt")
    caption = "Horses"
    tokens = tokenizer.tokenize(caption)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
    input_ids = np.array([input_ids], dtype=np.int64)  # Shape: (1, seq_len)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)  # Shape: (1, seq_len)
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)  # Shape: (1, seq_len)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[np.newaxis, :]  # Shape: (1, seq_len)
    
    # test
    input_ids_seq = input_ids[0]  # Remove batch dimension
    seq_len = input_ids.shape[1]
    left_idx = 0  # Index of [CLS]
    right_idx = seq_len - 1  # Index of [SEP]
    print("Input IDs:", input_ids_seq)
    print("Tokens:", [tokenizer.ids_to_tokens.get(id, '[UNK]') for id in input_ids_seq])

    # Special token IDs
    special_token_ids = [
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.mask_token_id
]

    text_token_mask = generate_text_token_mask(input_ids, special_token_ids)

    image = preprocess_image(".asset/demo7.jpg")

    # Prepare inputs
    inputs = {
        "img": image,
        "input_ids": input_ids,
        "attention_mask": attention_mask.astype(bool),
        "position_ids": position_ids,
        "token_type_ids": token_type_ids,
        "text_token_mask": text_token_mask,
    }

    # run inference
    logits, boxes = session.run(None, inputs)
    logits = np.squeeze(logits)  # Shape: (900, 256)
    boxes = np.squeeze(boxes)  # Shape: (900, 4)
    logits = 1 / (1 + np.exp(-logits))  # Shape: (900, 256)

    box_threshold = 0.25
    text_threshold = 0.3
    with_logits = True

    # Filter logits and boxes
    filt_mask = np.max(logits, axis=1) > box_threshold 
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    pred_phrases = []
    for idx, logit in enumerate(logits_filt):
        posmap = logit > text_threshold  # Shape: (256,)
        posmap = posmap.astype(bool)

        # Debugging statements
        print(f"\nPrediction {idx + 1}:")
        print(f"Max logit value: {np.max(logit):.4f}")
        print(f"posmap sum (number of positive positions): {np.sum(posmap)}")

        pred_phrase = get_phrases_from_posmap(posmap, {'input_ids': input_ids}, tokenizer, left_idx, right_idx)

        if pred_phrase == "":
            print("No phrase extracted from posmap.")
            continue

        print(f"Extracted phrase: {pred_phrase}")

        if with_logits:
            pred_phrases.append(f"{pred_phrase} ({np.max(logit):.4f})")
        else:
            pred_phrases.append(pred_phrase)

    IPython.embed()