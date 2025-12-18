## Overview

A Gradio-based web interface for an **AI-powered health information retrieval assistant**, designed to support nurses, physicians, and other hospital staff.

This project leverages **Retrieval-Augmented Generation (RAG)** to provide highly accurate and reliable question-answering capabilities.

## Few Test Set Cases
# 1. Direct Fact Retrieval (Accuracy Check)

These questions test if the system can find specific "needle in a haystack" data points.

- **Question:** What are the dimensions of the nodule found on Maria Gonzalez's thyroid?  
  **Expected Answer:** 1.8 cm x 1.2 cm

- **Question:** What specific food did Robert Hale eat that likely caused his hospital admission?  
  **Expected Answer:** Cured ham and soup (high-sodium meal at a wedding)

- **Question:** What is Liam Smith's temperature as measured at home?  
  **Expected Answer:** 102°F

- **Question:** Who referred Maria Gonzalez for her consultation?  
  **Expected Answer:** Dr. Sarah Lee (PCP)

---

# 2. Contextual Reasoning (Connecting the Dots)

These questions test if the system understands why something happened by linking two different parts of a document.

- **Question:** Why was Liam Smith prescribed Azithromycin instead of Penicillin or Amoxicillin?  
  **Testing:** Links the Allergies section (Penicillin → hives) to the Treatment Plan (due to penicillin allergy).

- **Question:** Why is Maria Gonzalez specifically concerned about medication safety?  
  **Testing:** Links Subjective history (trying to conceive) to the Plan/Reassurance.

- **Question:** Did Robert Hale’s diuretic dosage change during his hospital stay? If so, how?  
  **Testing:** Compares Hospital Course or Discharge Medications to the patient's implied previous history.  
  **Outcome:** Lasix increased to 40 mg daily (previously 20 mg).



## Demo

https://github.com/user-attachments/assets/a313f565-af80-4601-ada7-9b55303bfe2e

