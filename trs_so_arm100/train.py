from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from arm_env import ArmEnv
import gymnasium as gym
from tqdm import tqdm

# Configuración
NUM_ENVS = 4               # Entornos paralelos
TIMESTEPS = 2_000_000      # Duración del entrenamiento
SAVE_PATH = "arm_grasping" # Nombre del modelo guardado

# Crear entorno con Gymnasium
env = make_vec_env(lambda: gym.wrappers.RecordEpisodeStatistics(ArmEnv()), n_envs=NUM_ENVS)

# Definir una barra de progreso personalizada
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        """Inicializa la barra de progreso al inicio del entrenamiento."""
        self.pbar = tqdm(total=self.total_timesteps, desc="Entrenando", unit="step")

    def _on_step(self) -> bool:
        """Actualiza la barra de progreso en cada paso de entrenamiento."""
        self.pbar.update(self.model.n_envs)  # n_envs es el número de entornos paralelos
        return True

    def _on_training_end(self):
        """Cierra la barra de progreso al finalizar el entrenamiento."""
        if self.pbar is not None:
            self.pbar.close()

# Inicializar agente
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    policy_kwargs=dict(net_arch=[256, 256])
)

# Entrenar con barra de progreso
model.learn(total_timesteps=TIMESTEPS, callback=ProgressBarCallback(TIMESTEPS))
model.save(SAVE_PATH)
print(f"¡Entrenamiento completo! Modelo guardado en {SAVE_PATH}")

# Evaluar y visualizar el modelo entrenado
def evaluate():
    env = ArmEnv()
    obs, _ = env.reset()
    
    for _ in range(500):  # Ejecutar por 500 pasos
        action, _ = model.predict(obs, deterministic=True)  # Usar el modelo entrenado
        obs, reward, done, truncated, _ = env.step(action)
        env.render()  # Mostrar visualización en MuJoCo
        
        if done:
            obs, _ = env.reset()

    env.close()

print("Visualizando el modelo entrenado...")
evaluate()
