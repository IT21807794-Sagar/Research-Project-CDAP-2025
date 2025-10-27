
# response into structured format
def parse_question_response(response_text):
    question_data = {"question": "", "correct": "", "explanation": ""}

    lines = [line.strip() for line in response_text.split('\n') if line.strip()]

    for line in lines:
        if line.startswith("Question:"):
            question_data["question"] = line.replace("Question:", "").strip()
        elif line.startswith("Correct:"):
            question_data["correct"] = line.replace("Correct:", "").strip()
        elif line.startswith("Explanation:"):
            question_data["explanation"] = line.replace("Explanation:", "").strip()

    return question_data