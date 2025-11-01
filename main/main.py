from nesting.q4realbpp_loader import Q4RealBPPLoader
from packing_with_dqn_q4real import train_q4real_dqn_sequential, evaluate_q4real_dqn_sequential

instance = Q4RealBPPLoader.load_from_txt("nesting/3dBPP_6.txt")

# Train (CPU example); tweak episodes as you like
agent = train_q4real_dqn_sequential(instance, episodes=500, device="cuda", save_path=None)

# Evaluate
results = evaluate_q4real_dqn_sequential(agent, instance, num_eval=5, visualize=True)
print(results)
