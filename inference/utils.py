from transformers import AutoTokenizer
import tensorflow as tf

tokenizer_checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)


def check_similarity(trained_model, tokenizer, question1, question2, debug = 0):
  tokenizer_output = tokenizer(question1, question2, truncation=True, return_token_type_ids=True, max_length = 75, return_tensors = 'tf')
  logits = trained_model(**tokenizer_output)["logits"]
  predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
  if predicted_class_id == 1:
    if(debug):
        print("Both questions mean the same")
    return 1
  else:
    if(debug):
        print("Both the questions are different.")
    return 0