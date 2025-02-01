from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("./hindi_final.pt")
tokenizer = AutoTokenizer.from_pretrained("./hindi_tokenizer.json")
model.push_to_hub("raushankm/hindi_final")
tokenizer.push_to_hub("raushankm/hindi_tokenizer")