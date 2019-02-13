import matplotlib.pyplot as plt
from pandas import DataFrame
from math import sqrt

def exibir_fac_ruido(autocorrelacoes_ruido, qtd_ruidos_estimados):

    eixos = autocorrelacoes_ruido.plot(kind='bar', width=0.2, color='b',
                                        title='Autocorrelações dos Ruídos Estimados',
                                        ylim=(-1.0, 1.0), rot=0)
    eixos.set_xlabel('Lag', fontsize=12)
    eixos.set_ylabel('FAC', fontsize=12)
    eixos.axhline(color='000')

    eixos.axhline(2/sqrt(qtd_ruidos_estimados), color='r', linestyle='--')
    eixos.axhline(-2/sqrt(qtd_ruidos_estimados), color='r', linestyle='--')
    plt.show()

def comparacao_esperado_obtido(serie_original, modelo):

    intervalo_estimado = range(modelo.d + modelo.p + 1, serie_original.size + 1)
    comp_esperado_obtido = DataFrame(serie_original.loc[intervalo_estimado], columns=['Esperado'])
    comp_esperado_obtido.index = range(1, len(intervalo_estimado) + 1)

    comp_esperado_obtido['Obtido'] = comp_esperado_obtido['Esperado'] - modelo.ruidos_estimados

    eixos = comp_esperado_obtido.plot(title='Comparação entre o Esperado e o Obtido', color=['b', 'r'])
    eixos.set_xlabel('Tempo', fontsize=12)
    eixos.set_ylabel('Medição fenômeno', fontsize=12)
    plt.show()