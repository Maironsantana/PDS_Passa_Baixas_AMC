"""Experimento 1: reprodução didática do Butterworth passa-baixas do FCIAA."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter

from src.funcoes import (
    filtro_iir,
    calcular_resposta_frequencia,
    ganho_em_frequencia,
)


# A frequência de amostragem informa quantas amostras existem por segundo.
FS = 200_000  # Hz
# A frequência de corte separa aproximadamente a faixa preservada da atenuada.
FC = 15_600  # Hz
ORDEM = 15


def main():
    resultados = Path("results")
    resultados.mkdir(exist_ok=True)

    # Butterworth é um passa-baixas com resposta suave na faixa passante.

    b, a = butter(ORDEM, FC, btype="low", fs=FS)
    print("b =", b) # Coeficientes de entrada do filtro IIR
    print("a =", a) # Coeficientes de realimentação do filtro IIR


    ## ====== TESTE PRÁTICO ======
    # Criação de um sinal artificial de 10 milissegundos contendo várias senoides.
    duracao = 0.01  # 10 ms
    t = np.arange(0, duracao, 1 / FS)
    x = (
        np.sin(2 * np.pi * 5_000 * t)
        + np.sin(2 * np.pi * 10_000 * t)
        + np.sin(2 * np.pi * 30_000 * t)
        + np.sin(2 * np.pi * 60_000 * t)
    )

    # aplicando o filtro IIR ao sinal sintético.
    y = filtro_iir(x, b, a)

    # Visualizando o espectro do sinal antes e depois do filtro.
    # A FFT mostra como a energia do sinal se distribui entre frequências.
    espectro_x = np.fft.fft(x)
    espectro_y = np.fft.fft(y)

    # Mapeando apenas as frequências positivas para visualização no gráfico.
    frequencias_fft = np.fft.fftfreq(len(x), d=1 / FS)
    mascara_positiva = frequencias_fft >= 0

    #Gerando figura
    plt.figure(figsize=(9, 4.5))
    plt.plot(
        frequencias_fft[mascara_positiva] / 1_000,
        np.abs(espectro_x[mascara_positiva]),
        label="Antes do filtro",
        alpha=0.8,
    )
    plt.plot(
        frequencias_fft[mascara_positiva] / 1_000,
        np.abs(espectro_y[mascara_positiva]),
        label="Depois do filtro",
        alpha=0.8,
    )
    plt.axvline(FC / 1_000, color="tab:red", linestyle="--", label="Corte")
    plt.xlim(0, FS / 2_000)
    plt.xlabel("Frequência (kHz)")
    plt.ylabel("Magnitude da FFT")
    plt.title("Espectro do sinal sintético antes e depois")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(resultados / "espectro_sintetico_antes_depois.png", dpi=150)
    plt.close()

    print("\nFiguras salvas em results/.")


    ## ====== TESTE TEÓRICO ======
    #Para provar que o filtro realmente tem corte em 15,6kHz, calculamos a resposta em frequência.
    frequencias, ganho_db = calcular_resposta_frequencia(b, a, FS)
    
    # Imprime o ganho em dB para algumas frequências de interesse.
    for frequencia_alvo in (5_000, 10_000, FC, 30_000, 60_000):
        ganho = ganho_em_frequencia(frequencia_alvo, frequencias, ganho_db)
        print(f"Ganho em {frequencia_alvo / 1_000:g} kHz: {ganho:.2f} dB")

    # A frequência de corte de um Butterworth é esperada perto de -3 dB.
    ganho_corte = ganho_em_frequencia(FC, frequencias, ganho_db)
    print(f"\nValidação: ganho em 15,6 kHz = {ganho_corte:.2f} dB (esperado: ~ -3 dB)")

    plt.figure(figsize=(9, 4.5))
    plt.plot(frequencias / 1_000, ganho_db, color="tab:blue")
    plt.axvline(FC / 1_000, color="tab:red", linestyle="--", label="Corte: 15,6 kHz")
    plt.axhline(-3, color="gray", linestyle=":", label="-3 dB")
    plt.xlim(0, FS / 2_000)
    plt.ylim(-120, 5)
    plt.xlabel("Frequência (kHz)")
    plt.ylabel("Ganho (dB)")
    plt.title("Resposta em frequência do Butterworth passa-baixas")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(resultados / "resposta_frequencia.png", dpi=150)
    plt.close()

    


if __name__ == "__main__":
    main()
