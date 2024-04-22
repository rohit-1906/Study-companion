import os
import streamlit as st
from llm_functions import load_data, split_text, initialize_llm, generate_questions, create_retrieval_qa_chain

if 'questions' not in st.session_state:
    st.session_state['questions'] = 'empty'
    st.session_state['separated_question_list'] = 'empty'
    st.session_state['questions_to_answers'] = 'empty'
    st.session_state['submitted'] = 'empty'

st.title("Study companion")

openai_api_key = os.getenv("MY_OPENAI_KEY")

uploaded_file = st.file_uploader("Upload your study material", type=['pdf'])

if uploaded_file is not None:
    text_from_pdf = load_data(uploaded_file)

    doc_for_question_gen = split_text(text_from_pdf, chunk_size=10000, chunk_overlap=200)

    doc_for_question_answering = split_text(text_from_pdf, chunk_size=500, chunk_overlap=200)

    llm_question_gen = initialize_llm(openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k", temperature=0.4)

    llm_question_answering = initialize_llm(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.1)

    if st.session_state['questions'] == 'empty':
        with st.spinner("Generating questions..."):
            st.session_state['questions'] = generate_questions(llm=llm_question_gen, chain_type="refine", documents=doc_for_question_gen)

    if st.session_state['questions'] != 'empty':
        st.subheader("Generated Questions")
        st.info(st.session_state['questions'])

        st.session_state['q_list'] = st.session_state['questions'].split('\n')

        with st.form(key='my_form'):
            st.session_state['q_to_a'] = st.multiselect(label="Select questions to answer", options=st.session_state['q_list'])
            submitted = st.form_submit_button('Generate answers')
            if submitted:
                st.session_state['submitted'] = True

        if st.session_state['submitted']:
            with st.spinner("Generating answers..."):
                answer_chain = create_retrieval_qa_chain(openai_api_key=openai_api_key, documents=doc_for_question_answering, llm=llm_question_answering)
                for q in st.session_state['q_to_a']:
                    ans = answer_chain.run(q)
                    st.write(f"Question: {q}")
                    st.info(f"Answer: {ans}")

