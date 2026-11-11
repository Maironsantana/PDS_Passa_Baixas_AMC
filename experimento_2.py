"""Experimento 2: efeito do filtro em uma perturbação de sinal I/Q."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter

from src.funcoes import (
    aplicar_filtro_iir,
    calcular_ober,
    calcular_spr,
    carregar_amostra_radioml,
)


FS = 200_000  # Hz
FC = 15_600  # Hz
ORDEM = 15
CAMINHO_DATASET = Path("data/RML2016_10a_dict.pkl")


def espectro_centralizado(sinal):
    """Retorna a FFT com frequências negativas e positivas centralizadas."""
    espectro = np.fft.fft(sinal)
    frequencias = np.fft.fftfreq(len(sinal), d=1 / FS)
    return np.fft.fftshift(frequencias), np.fft.fftshift(espectro)


def main():
    resultados = Path("results")
    resultados.mkdir(exist_ok=True)

    b, a = butter(ORDEM, FC, btype="low", fs=FS)

    amostra = carregar_amostra_radioml(CAMINHO_DATASET, modulacao="QPSK", snr=18)
    # I é a componente em fase; Q é a componente em quadratura.
    i = amostra[0]
    q = amostra[1]
    x = i + 1j * q

    amostras = np.arange(len(x))
    plt.figure(figsize=(9, 4.5))
    plt.plot(amostras, i, label="I (em fase)")
    plt.plot(amostras, q, label="Q (em quadratura)")
    plt.xlabel("Índice da amostra")
    plt.ylabel("Amplitude")
    plt.title("Componentes I e Q de uma amostra QPSK")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(resultados / "iq_tempo.png", dpi=150)
    plt.close()

    # A semente fixa permite repetir exatamente a mesma perturbação.
    # 0,05 é uma suposição simples desta reprodução, não um valor do artigo.
    np.random.seed(0)
    delta = 0.05 * (
        np.random.normal(0, 1, len(x))
        + 1j * np.random.normal(0, 1, len(x))
    )

    # O filtro atua na perturbação, não no sinal I/Q original.
    delta_filtrada = (
        aplicar_filtro_iir(delta.real, b, a)
        + 1j * aplicar_filtro_iir(delta.imag, b, a)
    )
    x_adv = x + delta
    x_adv_filtrado = x + delta_filtrada

    # Para sinal complexo, a FFT possui conteúdo em frequências positivas e negativas.
    freq_delta, espectro_delta = espectro_centralizado(delta)
    freq_filtrada, espectro_filtrada = espectro_centralizado(delta_filtrada)

    plt.figure(figsize=(9, 4.5))
    plt.plot(freq_delta / 1_000, np.abs(espectro_delta), label="Perturbação original")
    plt.plot(
        freq_filtrada / 1_000,
        np.abs(espectro_filtrada),
        label="Perturbação filtrada",
    )
    plt.axvline(-FC / 1_000, color="tab:red", linestyle="--", label="Limites da banda")
    plt.axvline(FC / 1_000, color="tab:red", linestyle="--")
    plt.xlim(-FS / 2_000, FS / 2_000)
    plt.xlabel("Frequência (kHz)")
    plt.ylabel("Magnitude da FFT")
    plt.title("Energia da perturbação antes e depois do passa-baixas")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(resultados / "perturbacao_espectro_antes_depois.png", dpi=150)
    plt.close()

    # SPR compara energia do sinal com a energia adicionada pela perturbação.
    spr_sem_filtro = calcular_spr(x, x_adv)
    spr_com_filtro = calcular_spr(x, x_adv_filtrado)

    # OBER é a parcela da energia da perturbação que ficou fora de +/- 15,6 kHz.
    ober_sem_filtro = calcular_ober(delta, FS, FC)
    ober_com_filtro = calcular_ober(delta_filtrada, FS, FC)

    print(f"SPR sem filtro: {spr_sem_filtro:.2f} dB")
    print(f"SPR com filtro: {spr_com_filtro:.2f} dB")
    print()
    print(f"OBER sem filtro: {ober_sem_filtro:.2f} %")
    print(f"OBER com filtro: {ober_com_filtro:.2f} %")
    print()
    print("Valores reportados pelo FCIAA: SPR = 16,90 dB; OBER = 1,53 %")
    print("Esta reprodução usa ruído sintético; por isso os valores não precisam coincidir.")
    print("Figuras salvas em results/.")


if __name__ == "__main__":
    main()
