#!/usr/bin/env python3
"""
Quick test of the LLM judge function
"""

import subprocess

def evaluate_with_llm_judge(instruction: str, model_answer: str, ground_truth: str, model_name: str = "deepseek-r1") -> bool:
    """
    Use LLM-as-a-judge to evaluate if the model answer is correct.
    Returns True if the answer is judged as correct, False otherwise.
    """
    prompt = f"""You are an expert mathematics judge. Your task is to evaluate whether a model's answer to a math problem is correct.

Problem: {instruction}

Ground Truth Answer: {ground_truth}

Model's Answer: {model_answer}

Instructions:
1. Focus on the final numerical answer, not the reasoning process
2. Consider equivalent forms (e.g., 1.5 = 3/2 = 1 1/2)
3. Ignore minor formatting differences
4. Be strict about mathematical correctness
5. If the model's final answer matches the ground truth (even if expressed differently), judge as CORRECT
6. If the model's final answer is mathematically different from the ground truth, judge as INCORRECT

Respond with ONLY one word: either "CORRECT" or "INCORRECT"
"""

    try:
        # Use ollama to call deepseek-r1
        result = subprocess.run(
            ['ollama', 'run', model_name],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=30
        )
        
        if result.returncode == 0:
            response = result.stdout.strip().upper()
            print(f"LLM Response: '{response}'")
            # Check for exact match to avoid substring issues
            return response == "CORRECT" or "CORRECT" in response and "INCORRECT" not in response
        else:
            print(f"Error running ollama: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("LLM judge timeout - falling back to False")
        return False
    except Exception as e:
        print(f"Error in LLM judge: {e}")
        return False

if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            "instruction": "What is 2 + 3?",
            "model_answer": "The answer is 5",
            "ground_truth": "5",
            "expected": True
        },
        {
            "instruction": "What is 10 - 3?", 
            "model_answer": "Let me solve this step by step. 10 - 3 = 7. So the final answer is 7.",
            "ground_truth": "7",
            "expected": True
        },
        {
            "instruction": "What is 4 * 6?",
            "model_answer": "I think the answer is 25",
            "ground_truth": "24", 
            "expected": False  # We expect the judge to return False (incorrect answer)
        }
    ]
    
    print("Testing LLM Judge...")
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"Problem: {test['instruction']}")
        print(f"Model Answer: {test['model_answer']}")
        print(f"Ground Truth: {test['ground_truth']}")
        
        result = evaluate_with_llm_judge(test['instruction'], test['model_answer'], test['ground_truth'])
        
        print(f"LLM Judge Result: {result}")
        print(f"Expected: {test['expected']}")
        print(f"Correct: {'✓' if result == test['expected'] else '✗'}")
