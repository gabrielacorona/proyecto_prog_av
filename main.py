
# En equipos de 2 personas. Seran representados por RolA, RolB. 
# Hará un programa que represente la siguiente situación.

# Hay dos comensales y un cocinero. Solo hay un postre para 
# comer así que uno comerá y otro no. El cocinero tendrá listo el 
# postre para el primero que llegue a 20. cada uno incrementará el 
# contador en un numero aleatorio.

import tensorflow as tf
import random
import multiprocessing


def rolA():
     cont = 0
     while cont < 20:
          cont += random.random()
     
     return "termino A"

def rolB():
     cont = 0
     while cont < 20:
          cont += random.random()
     
     return "termino B"





def main():
     import keras
     procesoA = multiprocessing.Process(target=rolA)
     procesoB = multiprocessing.Process(target=rolA)
     procesoA.start()
     procesoB.start()

     procesoA.join()
     procesoB.join()
     print(procesoA,procesoB)


if __name__ == "__main__":
     main()