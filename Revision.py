import openai
import PyPDF2
import re
import os
from tkinter import Tk, filedialog
import key
import extract_analyze as ea
import parse_question_response as pq
import Check_validation

def select_pdf_file():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select PDF File",
        filetypes=[("PDF Files", "*.pdf")]
    )
    return file_path

# Identify main topics
def analyze_with_(text_content):
    try:
        content_sample = text_content[:3000]

        prompt = f"""
        Analyze the following text content and identify 3-5 main topics or subjects 
        that would be appropriate for creating short-answer questions. 
        Return them as a numbered list (1. Topic) without additional commentary.

        Text content:
        {content_sample}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200
        )

        topics = {}
        lines = [line.strip() for line in response.choices[0].message.content.split('\n') if line.strip()]
        for line in lines:
            if re.match(r'^\d+\.', line):
                parts = line.split('.', 1)
                if len(parts) == 2:
                    num = parts[0].strip()
                    topic = parts[1].strip()
                    topics[num] = topic

        return topics if topics else None

    except Exception as e:
        print(f"analysis error: {e}")
        return None

# Generate a question
def generate_question(subject, difficulty="medium"):
    prompt = f"""
    Create a {difficulty} difficulty short-answer question about {subject} with:
    1. A clear, specific question
    2. The correct answer
    3. A brief explanation

    Format exactly as:
    Question: [question text]
    Correct: [correct answer]
    Explanation: [explanation text]
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=250
        )
        return pq.parse_question_response(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"API error: {e}")
        return None

def display_question(question_data):
    print("\n" + "=" * 50)
    print("QUESTION:")
    print(question_data["question"])
    print("=" * 50)

# User's answer check
def check_answer(question_data, user_answer):
    correct = question_data["correct"].lower().strip()
    user_answer = user_answer.lower().strip()

    correct_terms = set(correct.split())
    user_terms = set(user_answer.split())
    match_score = len(correct_terms & user_terms) / len(correct_terms)

    if match_score > 0.5:  # 50% of key terms match
        print("\nThat's correct!")
        print(f"\nFULL ANSWER: {question_data['correct']}")
        print(f"\nEXPLANATION: {question_data['explanation']}")
        return True
    else:
        print("\nWrong. Here's more info:")
        print(f"\nEXPECTED ANSWER: {question_data['correct']}")
        print(f"\nEXPLANATION: {question_data['explanation']}")
        return False

def main():
    print("\n" + "=" * 50)
    print("Welcome")
    print("=" * 50 + "\n")

    pdf_path = select_pdf_file()
    if not pdf_path:
        print("No file selected. Exiting.")
        return

    print(f"\nSelected PDF: {os.path.basename(pdf_path)}")

    # Extract and analyze content
    with open(pdf_path, 'rb') as pdf_file:
        text_content = ea.extract_content_from_pdf(pdf_file)

    if not text_content:
        print("\nFailed to extract text from PDF.")
        return

    print("\nAnalyzing content...")
    topics = ea.analyze_content_for_topics(text_content)

    if not topics:
        print("\nCould not identify topics in the document.")
        return

    # Main interaction loop
    while True:
        print("\n" + "-" * 50)
        print("SELECT A TOPIC:")
        for num, topic in topics.items():
            print(f"{num}. {topic}")
        print("0. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "0":
            print("\nok!")
            break

        if choice in topics:
            selected_topic = topics[choice]
            print(f"\nSelected: {selected_topic}")

            difficulty = input("Choose difficulty (easy/medium/hard): ").lower()
            while difficulty not in ["easy", "medium", "hard"]:
                difficulty = input("Please enter easy/medium/hard: ").lower()


            while True:
                question_data = generate_question(selected_topic, difficulty)
                if not question_data:
                    print("Failed to generate question. Please try again.")
                    break

                display_question(question_data)

                user_answer = input("\nYour answer: ").strip()
                while not user_answer:
                    user_answer = input("Please enter an answer: ").strip()

                check_answer(question_data, user_answer)

                cont = input("\nAnother question on this topic? (y/n): ").lower()
                if cont != 'y':
                    break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
