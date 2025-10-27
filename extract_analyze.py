import PyPDF2
import Revision as Re
import Check_validation as CV

# Extract all text content
def extract_content_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        full_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        return full_text
    except Exception as e:
        print(f"\nError reading PDF: {str(e)}")
        return None


def analyze_content_for_topics(text_content):

    headings = CV.find_headings(text_content)

    if headings:
        return headings

    return Re.analyze_with_(text_content)