**********************************************************************************************************************
****************************PROYECTO FINAL ECONOMETRÍA III************************************************************
**CREATED: 26/02/2026
**AUTHOR: LUIS MAURICIO AGUIRRE STORNAIUOLO
clear all
set more off
cd "C:/Users/mauri/OneDrive/Documentos/Proyecto Econometria III"

import delimited using "EPEN 2024 BD_Publicacion Dpto (1).csv", clear varnames(1)


keep if c303 == 1 & c203 == 1
keep estrato c207 c208 c366 c376 c317 c310 c309_cod c318_t informal_p
keep if c208 <= 95 
keep if c318_t <= 112

********************************************************************************
cap destring c317, replace

recode c207 (1 = 1 "Hombre") (2 = 0 "Mujer"), gen("hombre")
gen mujer=(hombre==0)

recode informal_p (1 = 1 "Informal") (2 = 0 "Formal"),gen(informal)


capture drop macro_sector

recode c309_cod (100/399 = 1 "1_Agri_Pesca") ///
                (500/999 = 2 "2_Mineria") ///
                (1000/3399 = 3 "3_Manufac") ///
                (3500/3999 = 4 "4_Elec_Agua") ///
                (4100/4399 = 5 "5_Construc") ///
                (4500/4799 = 6 "6_Comercio") ///
                (4800/max = 7 "7_Servicios") ///
                (else = 8 "8_Otros"), gen(macro_sector)
		

recode c366 (1 2 = 1 "1_Sin_Nivel_Inicial") ///
            (3 4 = 2 "2_Primaria") ///
            (5/7 = 3 "3_Secundaria") ///
            (8 9 = 4 "4_Superior_Tecnica") ///
            (10 11 = 5 "5_Superior_Universitaria") ///
            (12 = 6 "6_Posgrado") ///
            (else = 1), gen(nivel_educativo)

recode c376 (10 = 1 "1_Castellano") ///
            (1/9 = 2 "2_Lengua_Nativa") ///
            (else = 3 "3_Otros"), gen(idioma_materno)
			
recode c310 (1 = 1 "1_Empleador") ///
            (2 = 2 "2_Independiente") ///
            (3 = 3 "3_Asalariado") ///
            (4 5 9 10 = 4 "4_Trabajador_Familiar") ///
            (6 = 5 "5_Trabajador_Hogar") ///
            (7 8 = 6 "6_Practicante") ///
            (else = 3), gen(categoria_ocupacional)

recode estrato (1 2 = 1 "1_Metropolitano") ///
               (3/5 = 2 "2_Urbano_Intermedio") ///
               (6/8 = 3 "3_Rural_Semi_Rural") ///
               (else = 1), gen(urbano)
			   
gen edad = c208
gen edad2 = edad^2
********************************************************************************
drop c309_cod c366 c376 c310 estrato
********************************************************************************
**Modelo inicial
logit informal edad edad2 c318_t mujer i.urbano i.idioma_materno ///
      ib5.nivel_educativo ib3.categoria_ocupacional i.macro_sector i.c317, robust

logit, or
*******************DIAGNÓSTICO DEL MODELO INICIAL*******************************
linktest
margins, dydx(*)

********************************************************************************
**MULTICOLINEALIDAD
********************************************************************************
reg informal edad edad2 c318_t mujer i.urbano i.idioma_materno ///
    i.nivel_educativo i.categoria_ocupacional i.macro_sector i.c317
vif //ENCONTRAMOS MULTICOLINEALIDAD ENTRE EDAD Y EDAD2
//Debido a que edad2 es una especificación polinómica de edad es normal tener un vif elevado y no afecta el modelo

********************************************************************************
**DESCARGAMOS LIBRERIAS PARA OBTENER LAS TABLAS DE RESULTADOS
**ESTAS TABLAS LAS USAREMOS PARA LA PRESENTACIÓN TIPO STREAMLIT MULTIPAGINA
ssc install outreg2
ssc install estout


* Regresión con errores robustos
logit informal edad edad2 c318_t mujer i.urbano i.idioma_materno ///
      ib5.nivel_educativo ib3.categoria_ocupacional i.macro_sector i.c317, robust

* Exportar Coeficientes
outreg2 using resultados_logit.xls, replace excel dec(4)

* Exportar Odds Ratios
logit, or
outreg2 using resultados_or.xls, replace excel dec(4) e(all)

*Efectos marginales
margins, dydx(*) post
outreg2 using efectos_marginales.xls, replace excel dec(4)

*Matriz de confusión de Stata
quietly logit informal edad edad2 c318_t mujer i.urbano i.idioma_materno ///
      ib5.nivel_educativo ib3.categoria_ocupacional i.macro_sector i.c317, robust

estat classification

*Grafico de Coeficientes
ssc install coefplot

coefplot, drop(_cons) xline(0)
graph export "grafico_coeficientes.png", replace


















