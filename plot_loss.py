import pandas as pd
import matplotlib.pyplot as plt

df_step = pd.read_csv("./checkpoint/logs/gpt_model_step_losses.csv")
df_eval = pd.read_csv("./checkpoint/logs/gpt_model_eval_losses.csv")

plt.plot(df_step["iter"], df_step["train_step_loss"])
plt.title("Pérdida de entrenamiento por iteración")
plt.xlabel("Iteración")
plt.ylabel("Pérdida")
plt.show()

plt.plot(df_eval["iter"], df_eval["train_eval_loss"], label="Entrenamiento (eval)")
plt.plot(df_eval["iter"], df_eval["val_eval_loss"], label="Val")
plt.legend()
plt.title("Pérdida de entrenamiento/validación (estimate_loss)")
plt.xlabel("Iteración")
plt.ylabel("Pérdida")
plt.show()
