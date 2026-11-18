from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RIDGE_FILE = (
    ROOT
    / "results"
    / "benchmark_audit"
    / "ridge_vs_zero_all_comparisons.csv"
)

ML_FILE = (
    ROOT
    / "results"
    / "ml_benchmark"
    / "dc_ml_nested_metrics.csv"
)

STABILITY_FILE = (
    ROOT
    / "results"
    / "diagnostics"
    / "dc_stability_summary.csv"
)

KEY_RESULTS_FILE = (
    ROOT
    / "results"
    / "article"
    / "updated_key_results.csv"
)

FIGURE_DIR = (
    ROOT
    / "results"
    / "figures"
)

ARTICLE_DIR = (
    ROOT
    / "results"
    / "article"
)

DIAGNOSTIC_DIR = (
    ROOT
    / "results"
    / "diagnostics"
)

GLOBAL_FIGURE = (
    FIGURE_DIR
    / "article_global_predictive_performance.png"
)

DC_FIGURE = (
    FIGURE_DIR
    / "article_dc_model_benchmark.png"
)

FINAL_TABLE = (
    ARTICLE_DIR
    / "article_final_key_results_table.csv"
)

STABILITY_TABLE = (
    ARTICLE_DIR
    / "article_dc_stability_table.csv"
)

INVENTORY_FILE = (
    ARTICLE_DIR
    / "article_final_tables_figures_inventory.csv"
)

CAPTIONS_FILE = (
    ARTICLE_DIR
    / "article_final_figure_captions.md"
)

DIAGNOSTIC_FILE = (
    DIAGNOSTIC_DIR
    / "article_final_tables_figures_diagnostic.csv"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARTICLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NOW = datetime.now().astimezone().isoformat()


EQUATION_LABELS = {
    "dA/dt": "dA/dt: exclusión escolar",
    "dC/dt": "dC/dt: logro adulto",
    "dF/dt": "dF/dt: financiamiento",
    "dL/dt": "dL/dt: alfabetización",
}

SAMPLE_LABELS = {
    "consecutive": "Transiciones anuales",
    "gap_max_2": "Brechas de hasta 2 años",
    "Ecuador annual series": "Ecuador 2009-2023",
}

MODEL_LABELS = {
    "ridge": "Ridge",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": (
        "HistGradientBoosting"
    ),
    "training_mean_baseline": (
        "Media de entrenamiento"
    ),
    "zero_baseline": "Cambio cero",
}


def save_figure(fig, path):
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def create_global_figure(ridge):
    order = [
        "dA/dt",
        "dC/dt",
        "dF/dt",
        "dL/dt",
    ]

    groups = [
        (
            "consecutive",
            "Global: transiciones anuales",
        ),
        (
            "gap_max_2",
            "Global: brechas de hasta 2 años",
        ),
        (
            "Ecuador annual series",
            "Ecuador: serie anual",
        ),
    ]

    x = np.arange(
        len(order)
    )

    width = 0.24

    fig, ax = plt.subplots(
        figsize=(12, 7),
    )

    colors = [
        "#2563eb",
        "#16a34a",
        "#f59e0b",
    ]

    for position, (
        sample,
        label,
    ) in enumerate(groups):
        subset = (
            ridge.loc[
                ridge[
                    "SAMPLE"
                ].eq(sample)
            ]
            .set_index("EQUATION")
            .reindex(order)
        )

        values = (
            subset[
                "RMSE_GAIN_PERCENT"
            ]
            .astype(float)
            .to_numpy()
        )

        offset = (
            position - 1
        ) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=colors[position],
        )

        for bar, value in zip(
            bars,
            values,
        ):
            vertical = (
                0.8
                if value >= 0
                else -1.5
            )

            alignment = (
                "bottom"
                if value >= 0
                else "top"
            )

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + vertical,
                f"{value:.1f}%",
                ha="center",
                va=alignment,
                fontsize=8,
            )

    ax.axhline(
        0,
        color="black",
        linewidth=1,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            EQUATION_LABELS[
                equation
            ]
            for equation in order
        ],
        rotation=12,
        ha="right",
    )

    ax.set_ylabel(
        "Mejora del RMSE frente a cambio cero (%)"
    )

    ax.set_title(
        "Rendimiento predictivo de Ridge frente "
        "a la referencia de cambio cero"
    )

    ax.legend(
        frameon=False,
        loc="lower left",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        GLOBAL_FIGURE,
    )


def create_dc_benchmark_figure(ml):
    models = [
        "ridge",
        "elastic_net",
        "random_forest",
        "training_mean_baseline",
        "zero_baseline",
        "hist_gradient_boosting",
    ]

    samples = [
        (
            "consecutive",
            "Transiciones anuales",
        ),
        (
            "gap_max_2",
            "Brechas de hasta 2 años",
        ),
    ]

    colors = {
        "consecutive": "#2563eb",
        "gap_max_2": "#16a34a",
    }

    x = np.arange(
        len(models)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(12, 7),
    )

    for position, (
        sample,
        label,
    ) in enumerate(samples):
        subset = (
            ml.loc[
                ml[
                    "SAMPLE"
                ].eq(sample)
            ]
            .set_index("MODEL")
            .reindex(models)
        )

        values = (
            subset[
                "RMSE_GAIN_PERCENT_VS_ZERO"
            ]
            .astype(float)
            .to_numpy()
        )

        offset = (
            position - 0.5
        ) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=colors[sample],
        )

        for bar, value in zip(
            bars,
            values,
        ):
            vertical = (
                0.18
                if value >= 0
                else -0.25
            )

            alignment = (
                "bottom"
                if value >= 0
                else "top"
            )

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + vertical,
                f"{value:.2f}%",
                ha="center",
                va=alignment,
                fontsize=8,
                rotation=90,
            )

    ax.axhline(
        0,
        color="black",
        linewidth=1,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            MODEL_LABELS[
                model
            ]
            for model in models
        ],
        rotation=18,
        ha="right",
    )

    ax.set_ylabel(
        "Mejora del RMSE frente a cambio cero (%)"
    )

    ax.set_title(
        "Comparación anidada de modelos para dC/dt"
    )

    ax.legend(
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        DC_FIGURE,
    )


def create_final_table(
    ridge,
    ml,
):
    ridge_table = (
        ridge[
            [
                "ANALYSIS_LEVEL",
                "SAMPLE",
                "EQUATION",
                "OBSERVATION_COUNT",
                "COUNTRY_COUNT",
                "RIDGE_RMSE",
                "ZERO_RMSE",
                "RMSE_GAIN_PERCENT",
                "IMPROVEMENT_CLASS",
            ]
        ]
        .copy()
    )

    ridge_table[
        "RESULT_TYPE"
    ] = "Ridge_vs_zero"

    ridge_table[
        "MODEL"
    ] = "Ridge"

    ridge_table = ridge_table.rename(
        columns={
            "RIDGE_RMSE": "MODEL_RMSE",
            "ZERO_RMSE": "REFERENCE_RMSE",
            "RMSE_GAIN_PERCENT": (
                "GAIN_PERCENT"
            ),
            "IMPROVEMENT_CLASS": (
                "CLASSIFICATION"
            ),
        }
    )

    nested_winners = []

    for sample in [
        "consecutive",
        "gap_max_2",
    ]:
        subset = ml.loc[
            ml["SAMPLE"].eq(sample)
        ].copy()

        candidates = subset.loc[
            ~subset[
                "MODEL"
            ].isin(
                [
                    "zero_baseline",
                    "training_mean_baseline",
                ]
            )
        ].copy()

        best = candidates.sort_values(
            "RMSE_CV"
        ).iloc[0]

        nested_winners.append(
            {
                "RESULT_TYPE": (
                    "Nested_ML_winner"
                ),
                "ANALYSIS_LEVEL": (
                    "Global nested GroupKFold"
                ),
                "SAMPLE": sample,
                "EQUATION": "dC/dt",
                "OBSERVATION_COUNT": int(
                    best[
                        "OBSERVATION_COUNT"
                    ]
                ),
                "COUNTRY_COUNT": int(
                    best[
                        "COUNTRY_COUNT"
                    ]
                ),
                "MODEL": MODEL_LABELS.get(
                    best["MODEL"],
                    best["MODEL"],
                ),
                "MODEL_RMSE": float(
                    best[
                        "RMSE_CV"
                    ]
                ),
                "REFERENCE_RMSE": float(
                    best[
                        "ZERO_RMSE_CV"
                    ]
                ),
                "GAIN_PERCENT": float(
                    best[
                        "RMSE_GAIN_PERCENT_VS_ZERO"
                    ]
                ),
                "CLASSIFICATION": (
                    "best_nested_model"
                ),
            }
        )

    nested_table = pd.DataFrame(
        nested_winners
    )

    final_table = pd.concat(
        [
            ridge_table[
                [
                    "RESULT_TYPE",
                    "ANALYSIS_LEVEL",
                    "SAMPLE",
                    "EQUATION",
                    "OBSERVATION_COUNT",
                    "COUNTRY_COUNT",
                    "MODEL",
                    "MODEL_RMSE",
                    "REFERENCE_RMSE",
                    "GAIN_PERCENT",
                    "CLASSIFICATION",
                ]
            ],
            nested_table,
        ],
        ignore_index=True,
    )

    final_table[
        "MODEL_RMSE"
    ] = final_table[
        "MODEL_RMSE"
    ].round(4)

    final_table[
        "REFERENCE_RMSE"
    ] = final_table[
        "REFERENCE_RMSE"
    ].round(4)

    final_table[
        "GAIN_PERCENT"
    ] = final_table[
        "GAIN_PERCENT"
    ].round(2)

    final_table.to_csv(
        FINAL_TABLE,
        index=False,
        encoding="utf-8-sig",
    )

    return final_table


def create_stability_table(stability):
    table = stability[
        [
            "SAMPLE",
            "OBSERVATION_COUNT",
            "COUNTRY_COUNT",
            "OVERALL_GAIN_PERCENT",
            "LOO_MIN_GAIN_PERCENT",
            "LOO_MAX_GAIN_PERCENT",
            "BOOTSTRAP_CI_2_5",
            "BOOTSTRAP_CI_97_5",
            "BOOTSTRAP_PROBABILITY_WIN",
            "DECADES_EVALUATED",
            "DECADES_WITH_POSITIVE_GAIN",
            "VALIDATION_STATUS",
        ]
    ].copy()

    percentage_columns = [
        "OVERALL_GAIN_PERCENT",
        "LOO_MIN_GAIN_PERCENT",
        "LOO_MAX_GAIN_PERCENT",
        "BOOTSTRAP_CI_2_5",
        "BOOTSTRAP_CI_97_5",
    ]

    for column in percentage_columns:
        table[column] = (
            table[column]
            .astype(float)
            .round(2)
        )

    table[
        "BOOTSTRAP_PROBABILITY_WIN"
    ] = (
        100
        * table[
            "BOOTSTRAP_PROBABILITY_WIN"
        ].astype(float)
    ).round(2)

    table.to_csv(
        STABILITY_TABLE,
        index=False,
        encoding="utf-8-sig",
    )

    return table


def create_inventory():
    rows = [
        {
            "ITEM_ID": "Figure_1",
            "ITEM_TYPE": "Figure",
            "FILE_PATH": (
                "results/figures/"
                "article_global_predictive_performance.png"
            ),
            "ARTICLE_SECTION": (
                "4. Resultados predictivos"
            ),
            "PURPOSE": (
                "Comparar la mejora de Ridge frente "
                "a cambio cero para las cuatro ecuaciones."
            ),
            "STATUS": "READY",
        },
        {
            "ITEM_ID": "Figure_2",
            "ITEM_TYPE": "Figure",
            "FILE_PATH": (
                "results/figures/"
                "article_dc_model_benchmark.png"
            ),
            "ARTICLE_SECTION": (
                "4. Resultados predictivos"
            ),
            "PURPOSE": (
                "Comparar modelos para dC/dt bajo "
                "validación anidada."
            ),
            "STATUS": "READY",
        },
        {
            "ITEM_ID": "Figure_3",
            "ITEM_TYPE": "Figure",
            "FILE_PATH": (
                "results/figures/"
                "aclf_ecuador_trajectories.png"
            ),
            "ARTICLE_SECTION": (
                "8. Caso de Ecuador"
            ),
            "PURPOSE": (
                "Mostrar las trayectorias ACLF observadas "
                "para Ecuador."
            ),
            "STATUS": "EXISTING_READY",
        },
        {
            "ITEM_ID": "Figure_4",
            "ITEM_TYPE": "Figure",
            "FILE_PATH": (
                "results/figures/"
                "ecuador_aclf_observed_predicted_rates.png"
            ),
            "ARTICLE_SECTION": (
                "8. Caso de Ecuador"
            ),
            "PURPOSE": (
                "Comparar tasas observadas y predichas "
                "para Ecuador."
            ),
            "STATUS": "EXISTING_READY",
        },
        {
            "ITEM_ID": "Table_1",
            "ITEM_TYPE": "Table",
            "FILE_PATH": (
                "results/article/"
                "master_selected_indicators.csv"
            ),
            "ARTICLE_SECTION": (
                "2. Datos y cobertura"
            ),
            "PURPOSE": (
                "Documentar los ocho indicadores "
                "educativos seleccionados."
            ),
            "STATUS": "EXISTING_READY",
        },
        {
            "ITEM_ID": "Table_2",
            "ITEM_TYPE": "Table",
            "FILE_PATH": (
                "results/article/"
                "article_final_key_results_table.csv"
            ),
            "ARTICLE_SECTION": (
                "4. Resultados predictivos"
            ),
            "PURPOSE": (
                "Resumir los resultados Ridge y los "
                "ganadores del benchmark anidado."
            ),
            "STATUS": "READY",
        },
        {
            "ITEM_ID": "Table_3",
            "ITEM_TYPE": "Table",
            "FILE_PATH": (
                "results/article/"
                "article_dc_stability_table.csv"
            ),
            "ARTICLE_SECTION": (
                "4. Resultados predictivos"
            ),
            "PURPOSE": (
                "Mostrar estabilidad leave-one-country-out, "
                "bootstrap y por décadas de dC/dt."
            ),
            "STATUS": "READY",
        },
        {
            "ITEM_ID": "Table_4",
            "ITEM_TYPE": "Table",
            "FILE_PATH": (
                "literature_review/evidence_synthesis/"
                "final_candidates_master_evidence_table.csv"
            ),
            "ARTICLE_SECTION": (
                "5. Revisión de literatura"
            ),
            "PURPOSE": (
                "Resumir los doce candidatos finales "
                "de la revisión."
            ),
            "STATUS": "EXISTING_READY",
        },
    ]

    inventory = pd.DataFrame(
        rows
    )

    inventory.to_csv(
        INVENTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return inventory


def create_captions():
    text = """# Leyendas propuestas para tablas y figuras

## Figura 1

**Rendimiento predictivo de Ridge frente a la referencia de cambio cero.** Los valores positivos indican una reducción del RMSE respecto de una predicción sin cambio. La ecuación de logro educativo adulto (dC/dt) fue la única que mejoró consistentemente la referencia en la muestra global y en el caso exploratorio de Ecuador.

## Figura 2

**Comparación de modelos para el cambio en logro educativo adulto.** Ridge y Elastic Net obtuvieron un rendimiento prácticamente equivalente. Los modelos no lineales no mostraron una ventaja consistente bajo validación cruzada agrupada por país.

## Figura 3

**Trayectorias de los estados ACLF en Ecuador, 2009-2023.** La figura presenta la evolución observada de exclusión escolar, logro educativo adulto, alfabetización juvenil y gasto público educativo.

## Figura 4

**Tasas observadas y predichas del sistema ACLF para Ecuador.** La comparación corresponde a validación leave-one-out sobre 14 transiciones anuales y debe interpretarse como evidencia exploratoria.

## Tabla 1

**Indicadores seleccionados para representar la dinámica educativa.** La tabla documenta la dimensión conceptual, dirección esperada, cobertura internacional y periodo disponible de cada indicador UNESCO UIS.

## Tabla 2

**Resultados predictivos principales.** La tabla compara Ridge con una referencia de cambio cero para las cuatro ecuaciones ACLF e incluye los modelos ganadores del benchmark anidado para dC/dt.

## Tabla 3

**Estabilidad de la señal predictiva de dC/dt.** Se presentan la mejora global, el rango leave-one-country-out, los intervalos bootstrap agrupados y la consistencia por décadas.

## Tabla 4

**Características metodológicas de los doce estudios incluidos en la síntesis final.** Los estudios se clasifican por diseño, fuente de datos, riesgo de sesgo y relevancia para el modelo de dinámica educativa.
"""

    CAPTIONS_FILE.write_text(
        text,
        encoding="utf-8",
    )


def main():
    required_files = [
        RIDGE_FILE,
        ML_FILE,
        STABILITY_FILE,
        KEY_RESULTS_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    ridge = pd.read_csv(
        RIDGE_FILE,
        encoding="utf-8-sig",
    )

    ml = pd.read_csv(
        ML_FILE,
        encoding="utf-8-sig",
    )

    stability = pd.read_csv(
        STABILITY_FILE,
        encoding="utf-8-sig",
    )

    key_results = pd.read_csv(
        KEY_RESULTS_FILE,
        encoding="utf-8-sig",
    )

    create_global_figure(
        ridge
    )

    create_dc_benchmark_figure(
        ml
    )

    final_table = create_final_table(
        ridge,
        ml,
    )

    stability_table = (
        create_stability_table(
            stability
        )
    )

    inventory = create_inventory()

    create_captions()

    expected_files = [
        GLOBAL_FIGURE,
        DC_FIGURE,
        FINAL_TABLE,
        STABILITY_TABLE,
        INVENTORY_FILE,
        CAPTIONS_FILE,
    ]

    files_ready = all(
        path.exists()
        and path.stat().st_size > 0
        for path in expected_files
    )

    global_dc_rows = ridge.loc[
        ridge[
            "EQUATION"
        ].eq("dC/dt")
        & ridge[
            "ANALYSIS_LEVEL"
        ].str.startswith(
            "Global",
            na=False,
        )
    ]

    valid = (
        len(ridge) == 12
        and len(ml) == 12
        and len(stability) == 2
        and len(key_results) == 14
        and len(final_table) == 14
        and len(stability_table) == 2
        and len(inventory) == 8
        and len(global_dc_rows) == 2
        and global_dc_rows[
            "RIDGE_BEATS_ZERO"
        ].astype(bool).all()
        and files_ready
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": NOW,
                "RIDGE_COMPARISONS": len(
                    ridge
                ),
                "ML_MODEL_RESULTS": len(
                    ml
                ),
                "STABILITY_ROWS": len(
                    stability
                ),
                "FINAL_TABLE_ROWS": len(
                    final_table
                ),
                "INVENTORY_ITEMS": len(
                    inventory
                ),
                "OUTPUT_FILES_READY": (
                    files_ready
                ),
                "VALIDATION_STATUS": (
                    "OK"
                    if valid
                    else "REVIEW"
                ),
            }
        ]
    )

    diagnostic.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== FIGURAS Y TABLAS "
        "FINALES DEL ARTÍCULO ====="
    )

    print(
        "FIGURA GLOBAL:",
        GLOBAL_FIGURE,
    )

    print(
        "FIGURA BENCHMARK DC:",
        DC_FIGURE,
    )

    print(
        "TABLA DE RESULTADOS:",
        FINAL_TABLE,
    )

    print(
        "TABLA DE ESTABILIDAD:",
        STABILITY_TABLE,
    )

    print(
        "INVENTARIO:",
        INVENTORY_FILE,
    )

    print(
        "LEYENDAS:",
        CAPTIONS_FILE,
    )

    print()
    print(
        "COMPARACIONES RIDGE:",
        len(ridge),
    )

    print(
        "RESULTADOS ML:",
        len(ml),
    )

    print(
        "FILAS TABLA FINAL:",
        len(final_table),
    )

    print(
        "ELEMENTOS INVENTARIADOS:",
        len(inventory),
    )

    print(
        "VALIDACION_STATUS:",
        diagnostic.loc[
            0,
            "VALIDATION_STATUS",
        ],
    )

    if not valid:
        raise ValueError(
            "Las figuras o tablas finales "
            "requieren revisión."
        )


if __name__ == "__main__":
    main()