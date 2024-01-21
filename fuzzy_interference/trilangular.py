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
smak['bardzo_dobry'] = fuzz.trimf(smak.universe, [7, 10, 10])
smak['dobry'] = fuzz.trimf(smak.universe, [5, 7, 9])
smak['umiarkowany'] = fuzz.trimf(smak.universe, [3, 5, 7])
smak['slaby'] = fuzz.trimf(smak.universe, [0, 3, 5])
smak['bardzo_slaby'] = fuzz.trimf(smak.universe, [0, 0, 2])

pikantnosc['niska'] = fuzz.trimf(pikantnosc.universe, [0, 2, 5])
pikantnosc['umiarkowana'] = fuzz.trimf(pikantnosc.universe, [5, 7, 9])
pikantnosc['wysoka'] = fuzz.trimf(pikantnosc.universe, [7, 10, 10])

konsystencja['nieidealna'] = fuzz.trimf(konsystencja.universe, [0, 2, 7])
konsystencja['idealna'] = fuzz.trimf(konsystencja.universe, [7, 10, 10])
konsystencja['troche_idealna'] = fuzz.trimf(konsystencja.universe, [5, 7, 9])

slodkosc['bardzo_slodka'] = fuzz.trimf(slodkosc.universe, [7, 10, 10])
slodkosc['slodka'] = fuzz.trimf(slodkosc.universe, [5, 7, 9])
slodkosc['umiarkowana'] = fuzz.trimf(slodkosc.universe, [3, 5, 7])
slodkosc['nieslodka'] = fuzz.trimf(slodkosc.universe, [0, 3, 5])
slodkosc['bardzo_nieslodka'] = fuzz.trimf(slodkosc.universe, [0, 0, 2])

przydatnosc['bardzo_przydatna'] = fuzz.trimf(przydatnosc.universe, [8, 10, 10])
przydatnosc['przydatna'] = fuzz.trimf(przydatnosc.universe, [5, 7, 9])
przydatnosc['srednio_przydatna'] = fuzz.trimf(przydatnosc.universe, [3, 5, 7])
przydatnosc['slabo_przydatna'] = fuzz.trimf(przydatnosc.universe, [0, 3, 5])
przydatnosc['nieprzydatna'] = fuzz.trimf(przydatnosc.universe, [0, 0, 2])


# Rysuj wykresy funkcji przynależności
fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, figsize=(8, 12))

ax0.plot(smak.universe, fuzz.trimf(smak.universe, [7, 10, 10]), 'r', linewidth=2, label='Bardzo Dobry')
ax0.plot(smak.universe, fuzz.trimf(smak.universe, [5, 7, 9]), 'g', linewidth=2, label='Dobry')
ax0.plot(smak.universe, fuzz.trimf(smak.universe, [3, 5, 7]), 'b', linewidth=2, label='Umiarkowany')
ax0.plot(smak.universe, fuzz.trimf(smak.universe, [0, 3, 5]), 'c', linewidth=2, label='Slaby')
ax0.plot(smak.universe, fuzz.trimf(smak.universe, [0, 0, 2]), 'm', linewidth=2, label='Bardzo Slaby')
ax0.set_title('Funkcje Przynależności - Smak')
ax0.legend()

ax1.plot(pikantnosc.universe, fuzz.trimf(pikantnosc.universe, [0, 2, 5]), 'r', linewidth=2, label='Niska')
ax1.plot(pikantnosc.universe, fuzz.trimf(pikantnosc.universe, [5, 7, 9]), 'g', linewidth=2, label='Umiarkowana')
ax1.plot(pikantnosc.universe, fuzz.trimf(pikantnosc.universe, [7, 10, 10]), 'b', linewidth=2, label='Wysoka')
ax1.set_title('Funkcje Przynależności - Pikantność')
ax1.legend()

ax2.plot(konsystencja.universe, fuzz.trimf(konsystencja.universe, [0, 2, 7]), 'r', linewidth=2, label='Nieidealna')
ax2.plot(konsystencja.universe, fuzz.trimf(konsystencja.universe, [5, 7, 9]), 'g', linewidth=2, label='Troche Idealna')
ax2.plot(konsystencja.universe, fuzz.trimf(konsystencja.universe, [7, 10, 10]), 'b', linewidth=2, label='Idealna')
ax2.set_title('Funkcje Przynależności - Konsystencja')
ax2.legend()

ax3.plot(slodkosc.universe, fuzz.trimf(slodkosc.universe, [0, 3, 5]), 'r', linewidth=2, label='Nieslodka')
ax3.plot(slodkosc.universe, fuzz.trimf(slodkosc.universe, [3, 5, 7]), 'g', linewidth=2, label='Umiarkowana')
ax3.plot(slodkosc.universe, fuzz.trimf(slodkosc.universe, [5, 7, 9]), 'b', linewidth=2, label='Slodka')
ax3.plot(slodkosc.universe, fuzz.trimf(slodkosc.universe, [7, 10, 10]), 'c', linewidth=2, label='Bardzo Slodka')
ax3.plot(slodkosc.universe, fuzz.trimf(slodkosc.universe, [0, 0, 2]), 'm', linewidth=2, label='Bardzo Nieslodka')
ax3.set_title('Funkcje Przynależności - Słodkość')
ax3.legend()

ax4.plot(przydatnosc.universe, fuzz.trimf(przydatnosc.universe, [0, 0, 2]), 'r', linewidth=2, label='Nie Przydatna')
ax4.plot(przydatnosc.universe, fuzz.trimf(przydatnosc.universe, [3, 5, 7]), 'g', linewidth=2, label='Słabo Przydatna')
ax4.plot(przydatnosc.universe, fuzz.trimf(przydatnosc.universe, [5, 7, 9]), 'b', linewidth=2, label='Średnio Przydatna')
ax4.plot(przydatnosc.universe, fuzz.trimf(przydatnosc.universe, [8, 10, 10]), 'c', linewidth=2, label='Przydatna')
ax4.plot(przydatnosc.universe, fuzz.trimf(przydatnosc.universe, [0, 0, 2]), 'm', linewidth=2, label='Bardzo Przydatna')
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