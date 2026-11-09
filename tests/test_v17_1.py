"""Testes da versão 17.1: mapas e prescrições preenchem o talhão até a linha do perímetro (sem faixa vazia)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
from shapely.geometry import box

from motor import book as bk
from motor.geo import ler_kml, montar_projeto
from motor.prescricao import _regiao_acima, superficies
from tests.test_mapas import _kml_perimetro, _kml_pontos, _laudo

RES = 20.0          # pixel grande: a faixa vazia antiga (meia a uma célula) ficaria bem medível


def _mapa():
    L = _laudo(20, com_2040=False)
    P = montar_projeto(L.dados, [ler_kml(_kml_perimetro(1), "Perimetro_TH_1.kml"),
                                 ler_kml(_kml_pontos(20), "Pontos_TH_1.kml")])
    return superficies(P, RES, modo="fiel")[0]


def test_desenho_do_mapa_passa_da_linha_do_perimetro():
    m = _mapa()
    x0, y0, x1, y1 = m.talhao.perimetro.bounds
    Zf, X, Y = bk._estendido(m, m.atributos["ph"])
    assert Zf.shape == (len(Y), len(X)) and np.isfinite(Zf).all()
    # curvas de contorno vão de centro a centro: o primeiro/último centro tem de ficar FORA do retângulo do talhão
    assert X[0] < x0 and X[-1] > x1 and Y[-1] < y0 and Y[0] > y1
    Zs, ext = bk._raster_suave(m, m.atributos["ph"])
    assert ext[0] < x0 and ext[1] > x1 and ext[2] < y0 and ext[3] > y1


def test_regiao_da_dose_vai_ate_a_borda():
    m = _mapa()
    g, per = m.grade, m.talhao.perimetro
    campo = np.broadcast_to(g.x[None, :], g.mascara.shape).astype(float)   # dose cresce para leste
    # limiar abaixo de tudo: a região cobre o talhão inteiro (antes parava no último centro de célula)
    assert _regiao_acima(g, campo, campo.min() - 0.5).contains(per)
    # limiar no meio: a parte leste do talhão inteira, encostada na linha (sem sobra para a dose vizinha)
    t = float(np.median(g.x))
    x0, y0, x1, y1 = per.bounds
    esperado = per.intersection(box(t, y0 - 1, x1 + 1, y1 + 1))
    obtido = _regiao_acima(g, campo, t).intersection(per)
    faltando = esperado.difference(obtido).area
    assert faltando < 0.05 * RES * (y1 - y0)            # antes: faixa de ~RES/2 ao longo da borda leste
