#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
} // la funcion tanh ya está definida en la biblioteca de matemáticas.

double sigmoid_prima(double x){
    return (exp(x)/((exp(x)+1)*(exp(x)+1)));
}  //esto es un puntero a alguna funcion definida luego

double (*g)(double x);  //esto es un puntero a alguna funcion definida luego
double (*g_prima)(double x);  //esto es un puntero a alguna funcion definida luego

/*
Parametros:
i = numero de patron que toca evaluar
Pattern = matriz de patrones
ocultas = vector de tamaño J que contiene el output de la neuronas ocultas
salida = vector de tamaño I que contiene el output de la neuronas de salida
w1 = pesos neuronas entrada-neuronas ocultas ()
w2 = pesos neuronas ocultas-neuronas salidas
PT = numero de patrones de entrenamiento
K = numero de neuronas de entrada en la capa 0, en realidad K+1: 0, 1, 2, …, K
I = numero de neuronas de salida en la capa 2: 1, 2, …, I
L = numero de patrones agrupados si se usa entrenamiento hibrido
J = numero de neuronas ocultas en la capa 1, en realidad J+1: 0, 1, 2, …, J
g = funcion de activacion
*/

void net_oculta(int i, int K, int J, double ** w1, double ** Pattern, double net_ocultas[]){
  for (int k = 1; k<=K; k++){
    for (int j = 1; j<J; j++ ){
          net_ocultas[j] += Pattern[i][k] * w1[k][j];
    }
  }
}


void y_oculta(int J, double y_oculta[],double net_oculta[], double (*g)(double x)){
  for (int i = 0; i < J; i++){
      y_oculta[i] = g(net_oculta[i]);
  }
}


 void net_salida(int I, int J, double ** w2, double net_salida[], double y_oculta[]){
  //neuronas ocultas
  for(int i=0; i<I; i++){
    for (int k=0; k<J; k++ ){
          net_salida[i] += w2[i][k] * y_oculta[k];
    }
  }
}


void y_salida(int I, double y_salida[],double net_salida[], double (*g)(double x)){
  for (int i = 0; i < I; i++){
      y_salida[i] = g(net_salida[i]);
  }
}


//funcion para calcular epsilon (error maximo permitido)
  double calculo_epsilon(int i, int I, double ** D, double y_salida[]){
  double error ;
  for(int k = 0; k < I; k++){
    error += 0.5 * pow((D[i][k]-y_salida[k]),2);
  }
  error = error / I;
  return error;
}

//funcion para calcular delta de capa de salida
//tengo duda de como invocar a sigmoide primo
void delta_salida(int i, int I, double ** D, double deltas_salida[],double net_salida[],double y_salida[], double (*g_prima)(double x) ){
  for(int j = 0; j < I; j++){
    deltas_salida[j] = (D[i][j]-y_salida[j]) * g_prima(net_salida[j]);
  }
}

//double * cambio_oculta()

// //correcion de pesos hidden-output
// double ** correcion_pesos_HO(int I, double eta, double * ocultas, double delta[I]){
//
// //se multiplican los vectores ocultas y delta y la matriz resultante se multiplia por eta
//
//
// }



int main(int argc, char **argv) {
   /* Se declaran las variables siguientes:
   I = numero de neuronas de salida en la capa 2: 1, 2, …, I
   J = numero de neuronas ocultas en la capa 1, en realidad J+1: 0, 1, 2, …, J
   K = numero de neuronas de entrada en la capa 0, en realidad K+1: 0, 1, 2, …, K
   L = numero de patrones agrupados si se usa entrenamiento hibrido
   Group = numero efectivo de patrones en cada grupo
   PT = numero de patrones de entrenamiento
   P = total de patrones en el archivo
   MaxEpocs = numero maximo de epocas de entrenamiento permitidas
   eta = tasa de aprendizaje
   alfa = momentum
   epsilon = error maximo permitido
   fractionTrainingPatterns = porcentaje de los patrones P usados para entrenamiento
   ActivationFunction = tipo de funcion de activacion: sigmoid, tanh
   Training = estrategia de entren   double ocultas[], y_oculta[J], net_salida[I], y_salida[I], error, delta_salida[I];amiento: continuous, batch, hybrid
   Pattern = arreglo para guardar los patrones en memoria
   D = arreglo para guardar las salidas esperadas en memoria

   ocultas = vector de tamaño J que contiene el output de la neuronas ocultas
   salida = vector de tamaño I que contiene el output de la neuronas de salida

   w1 = pesos neuronas entrada-neuronas ocultas (KxJ)
   w2 = pesos neuronas ocultas-neuronas salidas (JxI)
   w1v = w1 viejo
   w2v = w2 viejo
   */

   int I, J, K, L, P, PT, MaxEpocs, Group;
   int i, j, k, l, p, pt;  //contadores y subindices
   double eta, alfa, epsilon, fractionTrainingPatterns, azar;
   double Pattern[600][100], D[600][100];
   char symbol, ActivationFunction[20], Training[20], keyword[20], filename[50];


   double w1[100][100], w2[100][100], w1v[100][100], w2v[100][100];
   //vectores que contienen los outputs de la capa de salida y de la capa oculta
   double net_ocultas[J], y_oculta[J], net_salida[I], y_salida[I], delta_salida[I];
   FILE* fd;

   /* Se inicializa la semilla para numeros aleatorios (random seed) */
   srand ( time(NULL) );
   if (argc < 2) {
      printf("Usage: %s <archivo de datos>\n", argv[0]);
      exit(1);
   }
   strcpy(filename, argv[1]);
   printf("Se abre el archivo %s\n", filename);
   fd = fopen(filename,"r");

   printf("Inicia la lectura de parametros\n");
   /*keyword se lee con %s y representa la etiqueta que identifica al dato
   Los parametros estan separados de los datos por una linea que comienza con ----
   Se compara keyword con ---- para saber si ya terminaron de leerse los parametros.
   */
   fscanf(fd, "%s", keyword);
   while (strcmp(keyword, "----") != 0) {
          if (strcmp(keyword,"I") == 0) fscanf(fd, " = %d", &I);
     else if (strcmp(keyword,"J") == 0) fscanf(fd, " = %d", &J);
     else if (strcmp(keyword,"K") == 0) fscanf(fd, " = %d", &K);
     else if (strcmp(keyword,"L") == 0) fscanf(fd, " = %d", &L);
     else if (strcmp(keyword,"alfa") == 0) fscanf(fd, " = %lf", &alfa);
     else if (strcmp(keyword,"eta") == 0) fscanf(fd, " = %lf", &eta);
     else if (strcmp(keyword,"MaxEpocs") == 0) fscanf(fd, " = %d", &MaxEpocs);
     else if (strcmp(keyword,"epsilon") == 0) fscanf(fd, " = %lf", &epsilon);
     else if (strcmp(keyword,"fractionTrainingPatterns") == 0)
              fscanf(fd, " = %lf", &fractionTrainingPatterns);
     else if (strcmp(keyword,"ActivationFunction") == 0)
              fscanf(fd, " = %s", ActivationFunction);
     else if (strcmp(keyword,"Training") == 0) fscanf(fd, " = %s", Training);
     else {printf("ERROR: no se reconoce keyword = %s\n", keyword);
           perror("Keyword invalida");
           exit(1);
     }
     fscanf(fd, "%*[^\n]\n%s", keyword);
   }

printf("Parametros leidos. Ahora se leen y cuentan los patrones\n");
/* Se leen y cuentan los patrones. Cada patron posee K entradas en el archivo,
   y se agrega la entrada bias = 1 en la posicion 0 */
   P = 1;
   while (symbol=' ', fscanf(fd, "\n%*[^,],%c", &symbol) != EOF) {
    printf("P = %d, symbol=%c\n", P, symbol);
    Pattern[P][0] = 1.0;                 // bias = 1

    if (symbol == 'M')      {D[P][0] = 1; D[P][1] = 0;}   // cancer Maligno = (1, 0)
    else if (symbol == 'B') {D[P][0] = 0; D[P][1] = 1;}   // cancer Benigno = (0, 1)

    for (k=1; k<=K; k++)
        fscanf(fd, " , %lf", &Pattern[P][k]);
    P++;
   }

   PT = P * fractionTrainingPatterns;  // calcula el numero de patrones de entrenamiento

/*Se establece la función de activacion g(x)
  según indica el parámetro ActivationFunction y
  el numero de patrones en cada grupo de entrenamiento
*/
   if (strcmp(ActivationFunction,"sigmoid") == 0) g = sigmoid;
   else g = tanh;

   if (strcmp(ActivationFunction,"sigmoid") == 0) g_prima = sigmoid_prima;
   else g = tanh;

   if (strcmp(Training,"continuous") == 0) Group = 1;
   else if (strcmp(Training,"batch") == 0) Group = PT;
   else Group = L;

/* Se imprimen los parametros leidos y calculados para verificacion */
   printf("Neuronas de entrada K= %d\n", K);
   printf("Neuronas ocultas    J= %d\n", J);
   printf("Neurnas de salida   I= %d\n", I);
   printf("Numero de patrones agrupados L= %d\n", L);
   printf("Numero de patrones de entrenamiento PT= %d\n", PT);
   printf("Numero de patrones totales P= %d\n", P);
   printf("Maximo de epocas de entrenamiento permitidas MaxEpocs= %d\n", MaxEpocs);
   printf("Tasa de aprendizaje eta = %f\n", eta);
   printf("Momentum alfa = %f\n", alfa);
   printf("Error maximo permitido epsilon = %f\n", epsilon);
   printf("FractionTrainingPatterns = %f\n", fractionTrainingPatterns);
   printf("Funcion de activacion ActivationFunction= %s\n", ActivationFunction);
   printf("Estrategia de entrenamiento Training= %s\n\n", Training);

   for (int k = 1; k <= K; k++){
     for (int j = 1; j <= J; j++){
       w1[k][j] = azar = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0; //genera número aleatorio entre [-1, 1]
     }
   }

   for (int j = 1; j <= J; j++){
     for (int i = 1; i <= I; i++){
       w1[j][i] = azar = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0; //genera número aleatorio entre [-1, 1]
     }
   }


/*


   for (k=1; k<=K; k++){
     for (j=1; j<=J; j++)
        printf("%lf  ", w1[k][j]);
     printf("\n");
   }


   */

   //FORWARD PROPAGATION
   // for (i = 0; i < PT; i++){
   //   for (j = 0; j )
   // }

//Lo que sigue aguí puede eliminarse. Está solo para verificar que los
//patrones fueron leídos correctamente e ilustrar el llamado a g(x).

  //  printf("Los patrones leídos son:\n\n");
  //  for (p=0; p<P; p++) {
  //      printf("p=%d: D=(%f, %f) -> Pattern=",p, D[p][0], D[p][1]);
  //      for (k=0; k<=K; k++)
  //         printf("%lf  ", Pattern[p][k]);
  //      printf("\n");
  //  }
  //  printf("sigmoid(0.)=%lf, sigmoid(2.)=%lf\n", sigmoid(0.), sigmoid(2.0));
  //  printf("tanh(0.)=%lf, tanh(2.)=%lf\n", tanh(0.), tanh(2.0));
  //  printf("ActivationFunction = %s, g(0.)=%lf, g(2.)=%lf\n\n", ActivationFunction, g(0.), g(2.0));
   //
  //  printf("Se imprimen 10 numeros al azar entre -1 y 1\n");
  //  for (i=1; i<=10; i++) {
  //    azar = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0; //genera número aleatorio entre [-1, 1]
  //    printf("%f  ", azar);
  //  }
  //  printf("\n");
  //  exit(0);
}
