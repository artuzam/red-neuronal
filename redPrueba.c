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

double tanh_prima(double x){
    return (1-pow(tanh(x),2));
}  //esto es un puntero a alguna funcion definida luego

double (*g)(double x);  //esto es un puntero a alguna funcion definida luego
double (*g_prima)(double x);  //esto es un puntero a alguna funcion definida luego

// P normaliado = P / norma P
void normaliza(int P, int K, double Pattern[P][K]){
  double suma[P];
  double norma[P];
  for (int p=1; p<P; p++){
    for(int k=0; k<K; k++){
      suma[p] += pow(Pattern[p][k],2);
    }
  }
  for(int p=1; p<P; p++){
    norma[p] = sqrt(suma[p]);
  }
  for (int p=1; p<P; p++){
    for(int k=0; k<K; k++){
      if (norma != 0)
      Pattern[p][k] = Pattern[p][k] / norma[p];
      else{
        Pattern[p][k] = 0;
      }
    }
  }
}

void net_ocultaf(int i, int P, int K, int J, double w1[100][100], double Pattern[600][100], double net_oculta[J]){
printf("azar: %lf \n", w1[1][1]);
  for (int j = 1; j<J+1; j++){
    for (int k = 0; k<K; k++ ){
          net_oculta[j] += Pattern[i][k] * w1[k][j];
    }
  }
}



void y_ocultaf(int J, double y_oculta[J],double net_oculta[J], double (*g)(double x)){
  y_oculta[0] = 1;    //es el valor del bias
  for (int j = 1; j < J; j++){
      y_oculta[j] = g(net_oculta[j]);
  }
}


 void net_salidaf(int I, int J, double w2[200][100], double net_salida[I+1], double y_oculta[J+1]){
  //neuronas ocultas

  for(int i=1; i<I+1; i++){
    for (int j=0; j<J+1; j++ ){
          net_salida[i] += w2[j][i] * y_oculta[j];
    }
  }
}



void y_salidaf(int I, double y_salida[I+1],double net_salida[I+1], double (*g)(double x)){
  for (int i = 1; i < I+1; i++){
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
void delta_salidaf(int i, int I, double ** D, double delta_salida[],double net_salida[],double y_salida[], double (*g_prima)(double x) ){
  for(int j = 0; j < I; j++){
    delta_salida[j] = (D[i][j]-y_salida[j]) * g_prima(net_salida[j]);
  }
}

void delta_ocultaf(double delta_salida[], double ** w2, double y_oculta[], int J, int I, double delta_oculta[]){
  double suma;
  for (int j=0; j<J; j++){
    for(int i=0; i<I; i++){
      suma += (w2[j][i] * delta_salida[i]);
      delta_oculta[j] = suma * (y_oculta[j] * (1- y_oculta[j]));
    }
  }
}

// calcula y aplica cambios de pesos para neuronas en cualquier capa
// tam1 = tamaño capa A
// tam2 = tamaño capa B
// A -> B
void cambio_peso(double eta, double salida_capa[], double delta_capa[], int tam1, int tam2, double ** w, double ** cambio_w, double ** cambio_viejo){
  //calcula cambio
  for (int i=0; i<tam1; i++){
    for (int j=0; j<tam2; j++)
      cambio_w[i][j] = eta * delta_capa[j] * salida_capa[i];
  }
  //almacena pesos viejos
  cambio_viejo = w;
  //modifica w
  for (int i=0; i<tam1; i++){
    for (int j=0; j<tam2; j++)
      w[i][j] += cambio_w[i][j];
  }
}


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

   int I, J, K, L, P, PT, MaxEpocs, Group, value;
   int i, j, k, l, p, pt;  //contadores y subindices
   double eta, alfa, epsilon, fractionTrainingPatterns, azar;
   double Pattern[600][100], D[600][100];
   char symbol, ActivationFunction[20], Training[20], keyword[20], filename[50];


   double w1[100][100], w2[100][100], w1v[100][100], w2v[100][100], cambio_w1[100][100], cambio_w2[100][100];
   //vectores que contienen los outputs de la capa de salida y de la capa oculta
   double net_oculta[J+1], y_oculta[J+1], net_salida[I+1], y_salida[I+1], delta_salida[I+1], delta_oculta[J+1];
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
   while (fscanf(fd, "\n%*[^,],%d", &symbol) != EOF) {
    printf("P = %d, symbol=%lf\n", P, symbol);
    Pattern[P][0] = 1.0;                 // bias = 1

    if (symbol == 1)      {D[P][0] = 1;}   // cancer Maligno = (1, 0)
    else if (symbol == -1) {D[P][0] = -1;}   // cancer Benigno = (0, 1)

    for (k=1; k<=K; k++)
        fscanf(fd, " , %lf", &Pattern[P][k]);
    P++;
   }
   fscanf(fd, " , %lf", &value);

  PT = P * fractionTrainingPatterns;  // calcula el numero de patrones de entrenamiento

/*Se establece la función de activacion g(x)
  según indica el parámetro ActivationFunction y
  el numero de patrones en cada grupo de entrenamiento
*/
   if (strcmp(ActivationFunction,"sigmoid") == 0) g = sigmoid;
   else g = tanh;

   if (strcmp(ActivationFunction,"sigmoid") == 0) g_prima = sigmoid_prima;
   else g_prima = tanh_prima;

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

   for (int k = 0; k <= K+1; k++){
     for (int j = 0; j <= J+1; j++){
       w1[k][j] = azar = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0; //genera número aleatorio entre [-1, 1]
     }
   }

   for (int j = 1; j <= J+1; j++){
     for (int i = 1; i <= I+1; i++){
       w1[j][i] = azar = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0; //genera número aleatorio entre [-1, 1]
     }
   }

   for (p=0; p<K+1; p++) {
       for (k=0; k<=J+1; k++)
          printf("%lf  ", w1[p][k]);
       printf("\n");
   }

   printf("Los patrones leídos son:\n\n");
    for (p=0; p<P; p++) {
        printf("p=%d: D=(%f) -> Pattern=",p, D[p][0]);
        for (k=0; k<=K; k++)
           printf("%lf  ", Pattern[p][k]);
        printf("\n");
    }

   // normaliza(P, K, Pattern);
   //
   // printf("Los patrones normalizados son:\n\n");
   //  for (p=0; p<5 ; p++) {
   //      printf("p=%d: D=(%f, %f) -> Pattern=",p, D[p][0], D[p][1]);
   //      for (k=0; k<=K; k++)
   //         printf("%lf  ", Pattern[p][k]);
   //      printf("\n");
   //  }

   //prueba net_oculta
   //for(int i=0; i<; i++)
   i = 1;
   printf("azar: %lf \n", w1[1][1]);
   net_ocultaf(i, P, K, J, w1, Pattern, net_oculta);

   printf("NET_OCULTA");
   printf("\n");
   for(int i=1; i<J+1; i++){
     printf("%lf  ", net_oculta[i]);
   }
   printf("\n");


   //prueba y_oculta
   printf("Y_OCULTA");
   printf("\n");
   y_ocultaf(J, y_oculta, net_oculta, g);
   for(int i=1; i<J+1; i++){
     printf("%lf  ", y_oculta[i]);
   }

   //prueba net_salida
   net_salidaf(I, J, w2, net_salida, y_oculta);

   //prueba y_salida
   y_salidaf(I, y_salida,net_salida, g);

   printf("Y_SALIDA[1]: %lf \n", y_salida[1]);
   printf("Y_SALIDA[2]: %lf \n", y_salida[2]);
   printf("\n");


  //  //prueba y_oculta
  //  for(int i=0; i<5; i++)
  //   y_ocultaf(J, y_oculta, net_oculta, g);
   //
  //  printf("Y_OCULTA");
  //  printf("\n");
  //  for(int i=0; i<J; i++){
  //    printf("%lf  ", y_oculta[i]);
  //  }
   //
  //  //prueba net_salida
  //   net_salidaf(I, J, w2, net_salida, y_oculta);
   //
  //   printf("NET_SALIDA");
  //   printf("\n");
  //  for(int i=0; i<I; i++){
  //    printf("%lf  ", net_oculta[i]);
  //  }
   //
  //  //prueba y_salida
  //  y_salidaf(I, y_salida,net_salida, g);
   //
  //  printf("Y_SALIDA");
  //  printf("\n");
  //  for(int i=0; i<I; i++)
  //    printf("%lf  ", y_salida[i]);


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
