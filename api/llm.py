from transformers import pipeline

qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base"
)

def generate_answer(context, question):
    prompt = (
        "You are a helpful assistant. Use only the following context to answer the question.\n"
        "If the answer is not contained in the context, reply: 'Answer not found in the provided content.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    result = qa_pipeline(prompt, max_new_tokens=256)
    return result[0]["generated_text"].strip()
