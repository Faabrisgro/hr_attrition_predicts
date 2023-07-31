# **Elaniin | #1 GreatPlaceToWork | El Salvador 2024**
## Attrition Prediction Project by Fabrizio SgroIBM HR Dataset - EDA & Machine Learning Attrition Predictions
![portada](https://raw.githubusercontent.com/Faabrisgro/hr_attrition_predicts/master/portada.jpg)

# Análisis Exploratorio de Datos - IBM HR Attrition Prediction Dataset 

## Introducción

Este conjunto de datos contiene información anónima de colaboradores de una organización. El objetivo es predecir si un colaborador renunciará o no (variable Attrition). A continuación, se detallan las variables categóricas, ordinales y continuas presentes en el dataset.

## Variables del Dataset

### Categóricas:

- **`Attrition`**: Variable dicotómica. Indica 'Yes' (Si el colaborador renunció) o 'No' (si no renunció). Esta es la variable objetivo que deseamos **predecir**.

- `BusinessTravel`: Variable categórica. Especifica la frecuencia de viajes del colaborador (Travel_Rarely, Travel_Frequentl, Non-Travel).

- `Department`: Departamento en el que trabaja el colaborador (Research & Development, Sales, Human Resources).

- `Education`: Tipo de educación del colaborador. Variable categórica etiquetada: 1,2,3,4,5.
    + 1 'Below College'
    + 2 'College'
    + 3 'Bachelor'
    + 4 'Master'
    + 5 'Doctor'

- `EducationField`: Campo de educación. (Life Sciences, Medical, Marketing, Technical Degree, Other, Human Resources).

- `EnvironmentSatisfaction`: Variable ordinal (1,2,3,4) sobre la satisfacción con el ambiente laboral. Me interesa ver cuáles son los rangos de edad, departamento, funciones que menor satisfacción poseen y como afecta esto a su **posibilidad de renunciar**.
    + 1 'Low'
    + 2 'Medium'
    + 3 'High'
    + 4 'Very High'

- `Gender`: Género del colaborador, solo indican dos géneros de todos los que hay (Male, Female)

- `JobInvolvement`: Involucramiento del colaborador en los trabajos. Asigna 4 tipos de niveles, 1,2,3 y 4.
    + 1 'Low'
    + 2 'Medium'
    + 3 'High'
    + 4 'Very High' 

- `JobLevel`: Variable categórica, indica el nivel del colaborador, siendo 1 el más común y 5 el menos común:(1,2,3,4,5) 

- `JobRole`: Variable categórica, especifica el rol del colaborador:
    + Sales Executive	
    + Research Scientist
    + Laboratory Technician	
    + Manufacturing Director	
    + Healthcare Representative	
    + Manager	
    + Sales Representative	
    + Research Director
    + Human Resources

- `JobSatisfaction`: Nivel de satisfacción con el trabajo de cada colaborador.Va del 1-4.
    + 1 'Low'
    + 2 'Medium'
    + 3 'High'
    + 4 'Very High'

- `MaritalStatus`: Estado civil del colaborador. (Casado, Soltero o Divorciado)

- `OverTime`: Indica si el colaborador hace horas extras o no.

- `PerformanceRating`: Nivel de performance del colaborador: 
    + 1 'Low'
    + 2 'Good'
    + 3 'Excellent'
    + 4 'Outstanding'`

- `RelationShipSatisfaction`: 
    + 1 'Low'
    + 2 'Medium'
    + 3 'High'
    + 4 'Very High'

- `WorkLifeBalance`
    + 1 'Bad'
    + 2 'Good'
    + 3 'Better'
    + 4 'Best'

- `StockOptionLevel`: Variable categórica etiquetada con 0 (sin acciones), 1, 2 y 3 para indicar el nivel de Acciones que el colaborador recibe como compensación en caso de recibirlo. 


### Númericas:

- `Age`: Edad de los colaboradores.

- `DailyRate`: Compensación que gana el colaborador por día.

- `DistanceFromHome`: Distancia hasta el trabajo desde la casa del colaborador (Rango de 1-29, la unidad de medida no es específica).

- `EmployeeNumber`: ID Asignado al Colaborador, rango 1-1470. No creo que tenga relevancia ya que no tenemos forma de saber si el ID 1 = Al primer colaborador o al último.

- `HourlyRate`: Compensación por hora del colaborador.

- `MonthlyIncome`: Salario mensual del colaborador.

- `MonthlyRate`: Compensación mensual del colaborador.

- `NumCompaniesWorked`: Número de compañías previas en las que ha trabajdo el colaborador.

- `StandardHours`: Horas estándar. Variable númerica. Todos los colaboradores tienen 80.

- `TotalWorkingYears`: Cantidad de años total trabajando.

- `TrainingTimesLastYear`: Cantidad de veces que el colaborador recibió entrenamientos durante el año pasado.

- `YearsAtCompany`: Cantidad total de años trabajando en la organización.

- `YearsInCurrentRole`: Cantidad de años en el rol actual del colaborador.

- `YearsSinceLastPromotion`: Años dessde la última promoción.

- `YearsWithCurrManager`: Años con el actual manager.

## Contexto y Enunciación del Problema.

El análisis de HR Analytics nos ayuda a interpretar los datos organizacionales y descubrir tendencias relacionadas con las personas en la organización. En miras de lograr el objetivo de #1 Best Place To Work de El Salvador, deberemos incluir tecnología para ser capaces de detectar con mayor velocidad la posibilidad de attrition y desatisfacción de los colaboradores.

### Hipótesis 
Se cree que existen diferentes variables y condiciones que pueden favorecer a los colaboradores a renunciar e irse de las empresas. Este proyecto busca rechazar la hipótesis nula que declara que no hay posibilidad de decir con presición cuando una persona es factible a renunciar.

Para ello deberemos responder preguntas como:
+ ¿Existe alguna relación entre la posibilidad de renuncia y la cantidad de años con el mismo manager? 

+ ¿O tal vez, la cantidad de años en una misma posición? ¿Qué hay de los años sin una promoción?

+ ¿Hay departamentos con mayor rotación de personal que deban recibir mayor atención por parte del departamento de HR? 

+ ¿Qué tanto impacta la satisfacción con el ambiente en la posibilidad de renuncia? 

En este análisis, utilizaremos técnicas de exploración de datos para encontrar patrones o criterios que estén más relacionados con la atrición de colaboradores. El objetivo es identificar factores que puedan predecir posibles casos de atrición y ayudar al equipo de recursos humanos a tomar medidas adecuadas para retener a los empleados.
