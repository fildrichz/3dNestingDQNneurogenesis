from nesting.q4realbpp_loader import Q4RealBPPLoader
from packing_with_dqn_q4real import train_dqn_q4realbpp, evaluate_dqn_q4realbpp

instance = Q4RealBPPLoader.load_from_txt("nesting/3dBPP_11.txt")

# Train (CPU example); tweak episodes as you like
agent = train_dqn_q4realbpp(instance, episodes=200, device="cuda", save_path=None)

# Evaluate
results = evaluate_dqn_q4realbpp(agent, instance, num_eval=5, visualize=True)
print(results)
