from matplotlib import pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Zdefiniuj dane wejściowe (smak, pikantność, konsystencja, słodkość) i dane wyjściowe (przydatność)
smak = ctrl.Antecedent(np.arange(0, 11, 1), 'smak')
pikantnosc = ctrl.Antecedent(np.arange(0, 11, 1), 'pikantnosc')
konsystencja = ctrl.Antecedent(np.arange(0, 11, 1), 'konsystencja')
slodkosc = ctrl.Antecedent(np.arange(0, 11, 1), 'slodkosc')
przydatnosc = ctrl.Consequent(np.arange(0, 11, 1), 'przydatnosc')

# Zdefiniuj funkcje przynależności dla danych wejściowych i wyjściowych
smak['bardzo_dobry'] = fuzz.gaussmf(smak.universe, 10, 1)
smak['dobry'] = fuzz.gaussmf(smak.universe, 7, 1)
smak['umiarkowany'] = fuzz.gaussmf(smak.universe, 5, 1)
smak['slaby'] = fuzz.gaussmf(smak.universe, 3, 1)
smak['bardzo_slaby'] = fuzz.gaussmf(smak.universe, 0, 1)

pikantnosc['niska'] = fuzz.gaussmf(pikantnosc.universe, 2, 1)
pikantnosc['umiarkowana'] = fuzz.gaussmf(pikantnosc.universe, 7, 1)
pikantnosc['wysoka'] = fuzz.gaussmf(pikantnosc.universe, 10, 1)

konsystencja['nieidealna'] = fuzz.gaussmf(konsystencja.universe, 2, 1)
konsystencja['idealna'] = fuzz.gaussmf(konsystencja.universe, 10, 1)
konsystencja['troche_idealna'] = fuzz.gaussmf(konsystencja.universe, 7, 1)

slodkosc['bardzo_slodka'] = fuzz.gaussmf(slodkosc.universe, 10, 1)
slodkosc['slodka'] = fuzz.gaussmf(slodkosc.universe, 7, 1)
slodkosc['umiarkowana'] = fuzz.gaussmf(slodkosc.universe, 5, 1)
slodkosc['nieslodka'] = fuzz.gaussmf(slodkosc.universe, 3, 1)
slodkosc['bardzo_nieslodka'] = fuzz.gaussmf(slodkosc.universe, 0, 1)

przydatnosc['bardzo_przydatna'] = fuzz.gaussmf(przydatnosc.universe, 10, 1)
przydatnosc['przydatna'] = fuzz.gaussmf(przydatnosc.universe, 7, 1)
przydatnosc['srednio_przydatna'] = fuzz.gaussmf(przydatnosc.universe, 5, 1)
przydatnosc['slabo_przydatna'] = fuzz.gaussmf(przydatnosc.universe, 3, 1)
przydatnosc['nieprzydatna'] = fuzz.gaussmf(przydatnosc.universe, 0, 1)

# Rysuj wykresy funkcji przynależności
fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, figsize=(8, 12))

ax0.plot(smak.universe, fuzz.gaussmf(smak.universe, 10, 1), 'r', linewidth=2, label='Bardzo Dobry')
ax0.plot(smak.universe, fuzz.gaussmf(smak.universe, 7, 1), 'g', linewidth=2, label='Dobry')
ax0.plot(smak.universe, fuzz.gaussmf(smak.universe, 5, 1), 'b', linewidth=2, label='Umiarkowany')
ax0.plot(smak.universe, fuzz.gaussmf(smak.universe, 3, 1), 'c', linewidth=2, label='Slaby')
ax0.plot(smak.universe, fuzz.gaussmf(smak.universe, 0, 1), 'm', linewidth=2, label='Bardzo Slaby')
ax0.set_title('Funkcje Przynależności - Smak')
ax0.legend()

ax1.plot(pikantnosc.universe, fuzz.gaussmf(pikantnosc.universe, 2, 1), 'r', linewidth=2, label='Niska')
ax1.plot(pikantnosc.universe, fuzz.gaussmf(pikantnosc.universe, 7, 1), 'g', linewidth=2, label='Umiarkowana')
ax1.plot(pikantnosc.universe, fuzz.gaussmf(pikantnosc.universe, 10, 1), 'b', linewidth=2, label='Wysoka')
ax1.set_title('Funkcje Przynależności - Pikantność')
ax1.legend()

ax2.plot(konsystencja.universe, fuzz.gaussmf(konsystencja.universe, 2, 1), 'r', linewidth=2, label='Nieidealna')
ax2.plot(konsystencja.universe, fuzz.gaussmf(konsystencja.universe, 7, 1), 'g', linewidth=2, label='Troche Idealna')
ax2.plot(konsystencja.universe, fuzz.gaussmf(konsystencja.universe, 10, 1), 'b', linewidth=2, label='Idealna')
ax2.set_title('Funkcje Przynależności - Konsystencja')
ax2.legend()

ax3.plot(slodkosc.universe, fuzz.gaussmf(slodkosc.universe, 10, 1), 'r', linewidth=2, label='Bardzo Slodka')
ax3.plot(slodkosc.universe, fuzz.gaussmf(slodkosc.universe, 7, 1), 'g', linewidth=2, label='Slodka')
ax3.plot(slodkosc.universe, fuzz.gaussmf(slodkosc.universe, 5, 1), 'b', linewidth=2, label='Umiarkowana')
ax3.plot(slodkosc.universe, fuzz.gaussmf(slodkosc.universe, 3, 1), 'c', linewidth=2, label='Nieslodka')
ax3.plot(slodkosc.universe, fuzz.gaussmf(slodkosc.universe, 0, 1), 'm', linewidth=2, label='Bardzo Nieslodka')
ax3.set_title('Funkcje Przynależności - Słodkość')
ax3.legend()

ax4.plot(przydatnosc.universe, fuzz.gaussmf(przydatnosc.universe, 10, 1), 'r', linewidth=2, label='Bardzo Przydatna')
ax4.plot(przydatnosc.universe, fuzz.gaussmf(przydatnosc.universe, 7, 1), 'g', linewidth=2, label='Przydatna')
ax4.plot(przydatnosc.universe, fuzz.gaussmf(przydatnosc.universe, 5, 1), 'b', linewidth=2, label='Srednio Przydatna')
ax4.plot(przydatnosc.universe, fuzz.gaussmf(przydatnosc.universe, 3, 1), 'c', linewidth=2, label='Slabo Przydatna')
ax4.plot(przydatnosc.universe, fuzz.gaussmf(przydatnosc.universe, 0, 1), 'm', linewidth=2, label='Nieprzydatna')
ax4.set_title('Funkcje Przynależności - Przydatność')
ax4.legend()

plt.tight_layout()
plt.show()

# Zdefiniuj zasady opisujące ocenę przydatności potrawy
rule1 = ctrl.Rule(smak['bardzo_dobry'] & pikantnosc['niska'] & konsystencja['idealna'] & slodkosc['bardzo_slodka'], przydatnosc['bardzo_przydatna'])
rule2 = ctrl.Rule(smak['dobry'] & pikantnosc['niska'] & konsystencja['idealna'] & slodkosc['slodka'], przydatnosc['przydatna'])
rule3 = ctrl.Rule(smak['umiarkowany'] & pikantnosc['niska'] & konsystencja['troche_idealna'] & slodkosc['umiarkowana'], przydatnosc['srednio_przydatna'])
rule4 = ctrl.Rule(smak['slaby'] & pikantnosc['niska'] & konsystencja['nieidealna'] & slodkosc['nieslodka'], przydatnosc['slabo_przydatna'])
rule5 = ctrl.Rule(smak['bardzo_slaby'] & pikantnosc['niska'] & konsystencja['nieidealna'] & slodkosc['bardzo_nieslodka'], przydatnosc['nieprzydatna'])
rule6 = ctrl.Rule(smak['bardzo_dobry'] & pikantnosc['umiarkowana'] & konsystencja['idealna'] & slodkosc['bardzo_slodka'], przydatnosc['bardzo_przydatna'])
rule7 = ctrl.Rule(smak['dobry'] & pikantnosc['umiarkowana'] & konsystencja['idealna'] & slodkosc['slodka'], przydatnosc['przydatna'])
rule8 = ctrl.Rule(smak['umiarkowany'] & pikantnosc['umiarkowana'] & konsystencja['troche_idealna'] & slodkosc['umiarkowana'], przydatnosc['srednio_przydatna'])
rule9 = ctrl.Rule(smak['slaby'] & pikantnosc['umiarkowana'] & konsystencja['nieidealna'] & slodkosc['nieslodka'], przydatnosc['slabo_przydatna'])
rule10 = ctrl.Rule(smak['bardzo_slaby'] & pikantnosc['umiarkowana'] & konsystencja['nieidealna'] & slodkosc['bardzo_nieslodka'], przydatnosc['nieprzydatna'])
rule11 = ctrl.Rule(smak['bardzo_dobry'] & pikantnosc['wysoka'] & konsystencja['idealna'] & slodkosc['bardzo_slodka'], przydatnosc['bardzo_przydatna'])
rule12 = ctrl.Rule(smak['dobry'] & pikantnosc['wysoka'] & konsystencja['idealna'] & slodkosc['slodka'], przydatnosc['przydatna'])
rule13 = ctrl.Rule(smak['umiarkowany'] & pikantnosc['wysoka'] & konsystencja['troche_idealna'] & slodkosc['umiarkowana'], przydatnosc['srednio_przydatna'])
rule14 = ctrl.Rule(smak['slaby'] & pikantnosc['wysoka'] & konsystencja['nieidealna'] & slodkosc['nieslodka'], przydatnosc['slabo_przydatna'])
rule15 = ctrl.Rule(smak['bardzo_slaby'] & pikantnosc['wysoka'] & konsystencja['nieidealna'] & slodkosc['bardzo_nieslodka'], przydatnosc['nieprzydatna'])
rule16 = ctrl.Rule(smak['bardzo_dobry'] & pikantnosc['niska'] & konsystencja['troche_idealna'] & slodkosc['bardzo_slodka'], przydatnosc['bardzo_przydatna'])
rule17 = ctrl.Rule(smak['dobry'] & pikantnosc['niska'] & konsystencja['troche_idealna'] & slodkosc['slodka'], przydatnosc['przydatna'])
rule18 = ctrl.Rule(smak['umiarkowany'] & pikantnosc['niska'] & konsystencja['troche_idealna'] & slodkosc['umiarkowana'], przydatnosc['srednio_przydatna'])
rule19 = ctrl.Rule(smak['slaby'] & pikantnosc['niska'] & konsystencja['troche_idealna'] & slodkosc['nieslodka'], przydatnosc['slabo_przydatna'])
rule20 = ctrl.Rule(smak['bardzo_slaby'] & pikantnosc['niska'] & konsystencja['nieidealna'] & slodkosc['bardzo_nieslodka'], przydatnosc['nieprzydatna'])

# Stwórz system sterowania na podstawie zdefiniowanych reguł
system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20])
simulation = ctrl.ControlSystemSimulation(system)

# Zbieraj dane od użytkownika i dokonuj oceny przydatności potrawy
smak_input = float(input("Podaj smak (1-10): "))
pikantnosc_input = float(input("Podaj pikantność (1-10): "))
konsystencja_input = float(input("Podaj konsystencję (1-10): "))
slodkosc_input = float(input("Podaj słodkość (1-10): ")) 

# Przypisz wartości do zmiennych wejściowych
simulation.input['smak'] = smak_input
simulation.input['pikantnosc'] = pikantnosc_input
simulation.input['konsystencja'] = konsystencja_input
simulation.input['slodkosc'] = slodkosc_input

try:
    # Oblicz ocenę przydatności potrawy
    simulation.compute()
except ValueError as e:
    # Obsłuż błąd, jeśli nie udało się obliczyć przydatności potrawy
    print("Nie udało się określić przydatności potrawy")
else:
    # Wyświetl ocenę przydatności potrawy
    print("Ocena przydatności potrawy:", simulation.output['przydatnosc'])