from unsloth import check_python_modules, create_locked_down_function
from src.train.rl.env import GameBoard, execute_strategy
import numpy as np

global PRINTER
PRINTER = 0

# def extract_predicate(text):

# def extract_function(text):
#     if text.count("```") >= 2:
#         first = text.find("```") + 3
#         second = text.find("```", first)
#         fx = text[first:second].strip()
#         fx = fx.removeprefix("python\n")
#         fx = fx[fx.find("def") :]
#         if fx.startswith("def strategy(board):"):
#             return fx
#     return None


# def function_works(completions, **kwargs):
#     scores = []
#     for completion in completions:
#         score = 0
#         response = completion[0]["content"]
#         function = extract_function(response)
#         if function is not None:
#             ok, info = check_python_modules(function)
#         if function is None or "error" in info:
#             score = -2.0
#         else:
#             try:
#                 new_strategy = create_locked_down_function(function)
#                 score = 1.0
#             except:
#                 score = -0.5
#         scores.append(score)
#     return scores


# def no_cheating(completions, **kwargs):
#     scores = []
#     for completion in completions:
#         score = 0
#         response = completion[0]["content"]
#         function = extract_function(response)
#         if function is not None:
#             ok, info = check_python_modules(function)
#             scores.append(1.0 if ok else -20.0)  # Penalize heavily!
#         else:
#             scores.append(-1.0)  # Failed creating function
#     return scores


# def strategy_succeeds(completions, **kwargs):
#     global PRINTER
#     scores = []
#     # Generate a random game board with seed
#     seed = np.random.randint(10000)
#     for completion in completions:
#         printed = False
#         score = 0
#         response = completion[0]["content"]
#         function = extract_function(response)
#         if PRINTER % 5 == 0:
#             printed = True
#             print(function)
#         PRINTER += 1
#         if function is not None:
#             ok, info = check_python_modules(function)
#         if function is None or "error" in info:
#             scores.append(0)
#             continue
#         try:
#             new_strategy = create_locked_down_function(function)
#         except:
#             scores.append(0)
#             continue
#         try:
#             game = GameBoard(size=6, seed=seed, target=2048, probability_fours=0.10)
#             steps, game_state = execute_strategy(new_strategy, game)
#             print(f"Steps = {steps} State = {game_state}")
#             if printed is False:
#                 print(function)
#             print(game.board().pretty())
#             if game_state == "success":
#                 scores.append(20.0)  # Success - massively reward!
#             else:
#                 scores.append(2.0)  # Failed but function works!
#         except TimeoutError as e:
#             print("Timeout")
#             scores.append(-1.0)  # Failed with timeout
#         except Exception as e:
#             print(f"Exception = {str(e)}")
#             scores.append(-3.0)  # Failed
#     return scores
