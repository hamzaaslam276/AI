import streamlit as st
from utils import preprocess_image, extract_text, structure_data, generate_explanations
from PIL import Image
import tempfile

st.set_page_config(page_title="AI Medical Report Analyzer", layout="wide")

st.title("🩺 AI-Powered Medical Report Analyzer")
st.markdown("Upload a scanned medical report (image or PDF) to get understandable explanations of test results.")

uploaded_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Report", use_column_width=True)

    with st.spinner("Preprocessing image..."):
        processed_image = preprocess_image(image)

    with st.spinner("Extracting text..."):
        raw_text = extract_text(processed_image)

    with st.spinner("Structuring extracted data..."):
        structured_data = structure_data(raw_text)
        st.subheader("Extracted Test Data")
        st.dataframe(structured_data)

    with st.spinner("Generating AI Explanations..."):
        explanations = generate_explanations(structured_data)
        st.subheader("AI Explanations")
        for item in explanations:
            with st.expander(item['Test Name']):
                st.write(item['Explanation'])