from transformers import TFAutoModelForSequenceClassification

trained_model_checkpoint = '/media/muhammad_arham/F/Semester6/ML/Project/ServerAPI/pretrained_models/BERT/qqd/model'


qqd_model = TFAutoModelForSequenceClassification.from_pretrained(trained_model_checkpoint, num_labels=2)



