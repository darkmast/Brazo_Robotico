from stable_baselines3 import SAC
from arm_env import ArmEnv
import gymnasium as gym

# Cargar el modelo entrenado
MODEL_PATH = "arm_grasping.zip"
model = SAC.load(MODEL_PATH)

# Crear el entorno
env = ArmEnv()
obs, _ = env.reset()

print("Ejecutando el modelo entrenado...")

for _ in range(500):  # Ejecutar por 500 pasos
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    
    # Renderizar la simulación
    env.render()

    if done:
        obs, _ = env.reset()

env.close()
