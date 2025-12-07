import numpy as np                    # exponencial e faixa do gráfico
import matplotlib.pyplot as plt       # gráficos
from scipy.integrate import solve_ivp # raízes
from scipy.optimize import fsolve     # integração

"""PARÂMETROS DO PROCESSO (Tabela 1)"""
# Constantes cinéticas
k10 = 1.287e12  # h^-1
k20 = 1.287e12  # h^-1
k30 = 9.043e09  # L/(mol A.h)

# Termos de energia de ativação (-E/R) [K]
ER1 = -9758.3
ER2 = -9758.3
ER3 = -8560.0

# Calores de reação [kJ/mol] (já deu negativo)
dH_AB = -4.20 # mol de A
dH_BC = 11.00 # mol de B
dH_AD = 41.85 # mol de C

# Propriedades físicas e geométricas
rho = 0.9342    # kg/L
cp = 3.01       # kJ/(kg.K)
Kw = 4032.0     # kJ/(h.K.m^2)
Ar = 0.215      # m^2
V = 10.0        # L

# Condições de alimentação padrão
Ca0 = 5.10      # mol A/L
T0 = 130.0      # °C
Tk = 128.95     # ºC

def arrhenius(T_C, k0, ER_val):
  """Calcula k(T)
    T_C = temperatura em ºC, convertida para K
    k0 = constante cinética
    ER_val = -E/R
  """
  T_K = T_C + 273.15 # Passando para Kelvin
  return k0 * np.exp(ER_val / T_K)

def modelo_reator(t, y, u1, u2):
  """
  Sistema de EDOs do reator Van de Vusse.
  t = tempo em horas
  y = [Ca, Cb, T]
  u1 = F/V (taxa de diluição)
  u2 = Tk (temp da camisa)
  """
  Ca, Cb, T = y

  # Cinética dependente da temperatura (T atual do reator)
  k1 = arrhenius(T, k10, ER1)
  k2 = arrhenius(T, k20, ER2)
  k3 = arrhenius(T, k30, ER3)

  # Taxas de reação de acordo com as equações (3) e (4)
  r1 = k1 * Ca
  r2 = k2 * Cb
  r3 = k3 * (Ca**2)

  # Balanço de Massa A (equação 3)
  dCadt = u1 * (Ca0 - Ca) - r1 - r3

  # Balanço de Massa B (equação 4)
  dCbdt = -u1 * Cb + r1 - r2

  # Termo de troca térmica da equação 5
  termo_termico = (Kw * Ar / (rho * cp * V)) * (u2 - T)

  # Termo de alimentação da equação 5
  termo_alim = u1 * (T0 - T)

  # Termo de reação da equação 5
  termo_reacao = (1 / (rho * cp)) * (
    r1 * dH_AB +
    r2 * dH_BC +
    r3 * dH_AD
  )

  # Balanço de Energia (equação 5)
  dTdt = termo_reacao + termo_alim + termo_termico

  return [dCadt, dCbdt, dTdt]

def solve_estado_estacionario(u1_val, u2_val, initial_guess):
  """Resolve para o estado estacionário (derivadas = 0)"""

  def equations(y):
      # Passamos t=0, dT/dt = 0
      return modelo_reator(0, y, u1_val, u2_val)

  root = fsolve(equations, initial_guess)
  return root

"""ITEM A: Perfil Estacionário"""
# Faixa de F/V de 10 a 100 (como pedido)
fv_limites = np.linspace(10, 100, 100)
cb_estacionario = []

# Chute inicial para [Ca, Cb, T]
chute = [2.0, 1.0, 135.0]

for val_fv in fv_limites:
  # Resolvemos para cada F/V
  es = solve_estado_estacionario(val_fv, Tk, chute)
  cb_estacionario.append(es[1]) # Guarda o Cb calculado
  chute = es # Atualiza o chute para o próximo passo como o calculado

plt.figure(figsize=(8, 5))
plt.plot(fv_limites, cb_estacionario, linewidth=2, color='b')
plt.title(r'Perfil Estacionário de $C_B$ variando $F/V$', fontsize=18)
plt.xlabel(r'$F/V \ (h^{-1})$', fontsize=16)
plt.ylabel(r'$C_B \ (mol/L)$', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()

"""ITEM B: SIMULAÇÃO"""
#Condições para Mariana (nossa escolha)
u1_c1 = 40.0  # h^-1
tk_c1 = 128.9 # ºC (assumimos que foi dada na unidade errada)
u1_c2 = 75.0  # h^-1
tk_c2 = 128.9 # ºC (assumimos que foi dada na unidade errada)

condicoes = [
  {"nome": "CONDIÇÃO 1", "u1_base": u1_c1, "tk_base": tk_c1},
  {"nome": "CONDIÇÃO 2", "u1_base": u1_c2, "tk_base": tk_c2}
]

def simulacao(u1_base, tk_base, degrau):
  # 1. Estado estacionário na condição 1

  # Chute inicial para [Ca, Cb, T] (T em ºC)
  chute = [2.0, 1.0, 135.0]
  y_es = solve_estado_estacionario(u1_base, tk_base, chute)

  # 2. Aplicando degrau
  # (+2 e -2 a partir do EE calculado acima)
  u1_new = u1_base + degrau

  # 3. Simular Dinâmica
  tempo = (0, 1/6) # Simular por 10 minutos
  sol = solve_ivp( # Usando scipy
    fun=lambda t, y: modelo_reator(t, y, u1_new, tk_base),
    t_span=tempo,
    y0=y_es,
    method='LSODA',
    dense_output=True
  )

  return sol, y_es

# Plotagem para os degraus pedidos para a CONDIÇÃO 1 e 2
for cond in condicoes:
  # Iterando na lista de condições (1 e 2) de acordo com seus valores
  # degrau +2
  sol_pos, es_positivo = simulacao(cond['u1_base'], cond['tk_base'], +2.0)

  # degrau -2
  sol_neg, es_negativo = simulacao(cond['u1_base'], cond['tk_base'], -2.0)
  
  # Plotagem
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
  fig.suptitle(f"Dinâmica do Reator - {cond['nome']} ($u_1$ = {cond['u1_base']} $h^{{-1}}$)", fontsize=16)
  tempo_neg_min = sol_neg.t * 60
  tempo_pos_min = sol_pos.t * 60

  # Gráfico de Ca
  ax1.plot(tempo_pos_min, sol_pos.y[0], 'r-')
  ax1.plot(tempo_neg_min, sol_neg.y[0], 'b-')
  ax1.axhline(y=es_positivo[0], color='k', linestyle='--', alpha=0.5)
  ax1.set_xlabel('Tempo (min)', fontsize=14)
  ax1.set_ylabel('Concentração Ca (mol/L)', fontsize=14)
  ax1.set_title('Resposta de Ca', fontsize=14)
  ax1.tick_params(axis='x', labelsize=14)
  ax1.tick_params(axis='y', labelsize=14)
  ax1.grid(True)

  # Gráfico de Cb
  ax2.plot(tempo_pos_min, sol_pos.y[1], 'r-')
  ax2.plot(tempo_neg_min, sol_neg.y[1], 'b-')
  ax2.axhline(y=es_positivo[1], color='k', linestyle='--', alpha=0.5)
  ax2.set_xlabel('Tempo (min)', fontsize=14)
  ax2.set_ylabel('Concentração Cb (mol/L)', fontsize=14)
  ax2.set_title('Resposta de Cb', fontsize=14)
  ax2.tick_params(axis='x', labelsize=14)
  ax2.tick_params(axis='y', labelsize=14)
  ax2.grid(True)

  # Gráfico de T
  ax3.plot(tempo_pos_min, sol_pos.y[2], 'r-', label=f'Degrau +2 \n($u_1$ = {cond["u1_base"]+2} $h^{{-1}}$)')
  ax3.plot(tempo_neg_min, sol_neg.y[2], 'b-', label=f'Degrau -2 \n($u_1$ = {cond["u1_base"]-2} $h^{{-1}}$)')
  ax3.axhline(y=es_positivo[2], color='k', linestyle='--', alpha=0.5, label='EE Inicial')
  ax3.set_xlabel('Tempo (min)', fontsize=14)
  ax3.set_ylabel('Temperatura (ºC)', fontsize=14)
  ax3.set_title('Resposta de Temperatura', fontsize=14)
  ax3.grid(True)
  ax3.tick_params(axis='x', labelsize=14)
  ax3.tick_params(axis='y', labelsize=14)
  ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
  
  plt.tight_layout()
  plt.show()