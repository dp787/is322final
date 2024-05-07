from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import torch 

# Initialize the tokenizer, retriever, and generator
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact", use_dummy_dataset=True)
generator = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")

def generate_with_rag(prompt):
    # Retrieve relevant documents
    retriever_output = retriever.retrieve(prompt)

    # Select the top-k retrieved documents
    relevant_docs = [doc['text'] for doc in retriever_output]

    # Format the input with retrieved documents
    input_text = f"{prompt} Relevant Documents: {' '.join(relevant_docs)}"

    # Tokenize the input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(generator.device)

    # Generate response
    generated = generator.generate(input_ids=input_ids)

    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Modify your main function to use generate_with_rag
def main():
    st.set_page_config(page_title="img 2 audio story", page_icon="🤖")

    st.header("Turn img into audio story")
    uploaded_file=st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        bytes_data=uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        scenario=img2text(uploaded_file.name)
        story=generate_with_rag(scenario)  # Generate using RAG
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.flac")

if __name__=='__main__':
    main()
