---
license: other
language:
- en
widget:
- text: >
    '[CLS]\nQuestion: Are there any pedestrians crossing the road? If yes, how
    many?\nAnswer: 1\nStudent: One'
  example_title: Counting
tags:
- vision-language
- autonomous-driving
---

### What is this?
Lingo-Judge, a novel evaluation metric that aligns closely with human judgment on the LingoQA evaluation suite.

### How to use
For LingoQA datasets and benchmark, please see https://github.com/wayveai/LingoQA

Try out this simple example of using Lingo-Judge:
```python
# Import necessary libraries
from transformers import pipeline

# Define the model name to be used in the pipeline
model_name = 'wayveai/Lingo-Judge'

# Define the question and its corresponding answer and prediction
question = "Are there any pedestrians crossing the road? If yes, how many?"
answer = "1"
prediction = "Yes, there is one"

# Initialize the pipeline with the specified model, device, and other parameters
pipe = pipeline("text-classification", model=model_name)
# Format the input string with the question, answer, and prediction
input = f"[CLS]\nQuestion: {question}\nAnswer: {answer}\nStudent: {prediction}"

# Pass the input through the pipeline to get the result
result = pipe(input)

# Print the result and score
score = result[0]['score']
print(score > 0.5, score)
```


### Citation
If you find our work useful in your research, please consider citing:

```bibtex
@article{marcu2023lingoqa,
  title={LingoQA: Video Question Answering for Autonomous Driving}, 
  author={Ana-Maria Marcu and Long Chen and Jan Hünermann and Alice Karnsund and Benoit Hanotte and Prajwal Chidananda and Saurabh Nair and Vijay Badrinarayanan and Alex Kendall and Jamie Shotton and Oleg Sinavski},
  journal={arXiv preprint arXiv:2312.14115},
  year={2023},
}
```