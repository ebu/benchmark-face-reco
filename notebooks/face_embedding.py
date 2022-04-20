
def face_embedding (model_emb,faces_resize_np):
    embeddings_np = model_emb.predict_on_batch(faces_resize_np)
    return(embeddings_np)
