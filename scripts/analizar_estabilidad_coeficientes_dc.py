from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from validar_temporalmente_dc import (
    FEATURES,
    TARGET,
    TEMPORAL_ORIGINS,
    TEST_WINDOW_YEARS,
    choose_alpha_by_country,
    fit_ridge,
    prepare_matrices,
)


INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "uis_model_aclf_rates.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "coefficient_stability"
)

COEFFICIENTS_FILE = (
    OUTPUT_DIR
    / "dc_temporal_coefficient_stability.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "dc_temporal_coefficient_sign_summary.csv"
)

DIAGNOSTIC_FILE = (
    ROOT
    / "results"
    / "diagnostics"
    / "dc_temporal_coefficient_stability_diagnostic.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

NOW = datetime.now().astimezone().isoformat()

EXPECTED_SIGNS = {
    "PREVIOUS_A": "negative",
    "PREVIOUS_C": "negative",
    "PREVIOUS_L": "positive",
    "PREVIOUS_F": "positive",
}


def coefficient_sign(value):
    if value > 0:
        return "positive"

    if value < 0:
        return "negative"

    return "zero"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            INPUT_FILE
        )

    data = pd.read_parquet(
        INPUT_FILE
    )

    required_columns = (
        FEATURES
        + [
            TARGET,
            "COUNTRY_ID",
            "YEAR",
            "DELTA_T",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas: {missing_columns}"
        )

    samples = {
        "consecutive": (
            data.loc[
                data["DELTA_T"].eq(1)
            ].copy()
        ),
        "gap_max_2": (
            data.loc[
                data["DELTA_T"].le(2)
            ].copy()
        ),
    }

    coefficient_rows = []

    for sample_name, sample in samples.items():
        sample = (
            sample
            .dropna(
                subset=[
                    TARGET,
                    "YEAR",
                    "COUNTRY_ID",
                ]
            )
            .sort_values(
                [
                    "YEAR",
                    "COUNTRY_ID",
                ]
            )
            .reset_index(drop=True)
        )

        for origin in TEMPORAL_ORIGINS:
            test_end = (
                origin
                + TEST_WINDOW_YEARS
            )

            train = sample.loc[
                sample["YEAR"].le(origin)
            ].copy()

            test = sample.loc[
                sample["YEAR"].gt(origin)
                & sample["YEAR"].le(
                    test_end
                )
            ].copy()

            if (
                len(train) < 40
                or len(test) < 10
                or train[
                    "COUNTRY_ID"
                ].nunique() < 5
            ):
                print(
                    sample_name,
                    origin,
                    "OMITIDO:",
                    "train=",
                    len(train),
                    "test=",
                    len(test),
                )
                continue

            if int(test["YEAR"].min()) <= int(
                train["YEAR"].max()
            ):
                raise ValueError(
                    "Se detectó fuga temporal."
                )

            best_alpha = (
                choose_alpha_by_country(
                    train
                )
            )

            train_x, _ = prepare_matrices(
                train,
                test
            )

            train_y = (
                train[TARGET]
                .to_numpy(dtype=float)
            )

            (
                target_mean,
                coefficients,
            ) = fit_ridge(
                train_x,
                train_y,
                best_alpha,
            )

            coefficient_rows.append(
                {
                    "SAMPLE": sample_name,
                    "TEMPORAL_ORIGIN": origin,
                    "TEST_START_YEAR": int(
                        test["YEAR"].min()
                    ),
                    "TEST_END_YEAR": int(
                        test["YEAR"].max()
                    ),
                    "TRAIN_OBSERVATIONS": len(
                        train
                    ),
                    "TRAIN_COUNTRIES": int(
                        train[
                            "COUNTRY_ID"
                        ].nunique()
                    ),
                    "BEST_ALPHA": best_alpha,
                    "TERM": "INTERCEPT",
                    "STANDARDIZED_COEFFICIENT": (
                        target_mean
                    ),
                    "SIGN": coefficient_sign(
                        target_mean
                    ),
                    "EXPECTED_SIGN": (
                        "not_applicable"
                    ),
                    "SIGN_MATCHES_EXPECTED": (
                        True
                    ),
                }
            )

            for feature, coefficient in zip(
                FEATURES,
                coefficients,
            ):
                observed_sign = (
                    coefficient_sign(
                        float(coefficient)
                    )
                )

                expected_sign = (
                    EXPECTED_SIGNS[
                        feature
                    ]
                )

                coefficient_rows.append(
                    {
                        "SAMPLE": sample_name,
                        "TEMPORAL_ORIGIN": (
                            origin
                        ),
                        "TEST_START_YEAR": int(
                            test["YEAR"].min()
                        ),
                        "TEST_END_YEAR": int(
                            test["YEAR"].max()
                        ),
                        "TRAIN_OBSERVATIONS": (
                            len(train)
                        ),
                        "TRAIN_COUNTRIES": int(
                            train[
                                "COUNTRY_ID"
                            ].nunique()
                        ),
                        "BEST_ALPHA": (
                            best_alpha
                        ),
                        "TERM": feature,
                        "STANDARDIZED_COEFFICIENT": float(
                            coefficient
                        ),
                        "SIGN": observed_sign,
                        "EXPECTED_SIGN": (
                            expected_sign
                        ),
                        "SIGN_MATCHES_EXPECTED": bool(
                            observed_sign
                            == expected_sign
                        ),
                    }
                )

    coefficients = pd.DataFrame(
        coefficient_rows
    )

    if coefficients.empty:
        raise ValueError(
            "No se generaron coeficientes."
        )

    coefficients.to_csv(
        COEFFICIENTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    predictor_coefficients = (
        coefficients.loc[
            coefficients[
                "TERM"
            ].isin(FEATURES)
        ].copy()
    )

    summary_rows = []

    for feature in FEATURES:
        subset = (
            predictor_coefficients.loc[
                predictor_coefficients[
                    "TERM"
                ].eq(feature)
            ].copy()
        )

        sign_counts = (
            subset["SIGN"]
            .value_counts()
            .to_dict()
        )

        expected_sign = (
            EXPECTED_SIGNS[
                feature
            ]
        )

        matching_count = int(
            subset[
                "SIGN_MATCHES_EXPECTED"
            ].sum()
        )

        total_count = len(subset)

        consistency_percent = float(
            100
            * matching_count
            / total_count
        )

        summary_rows.append(
            {
                "TERM": feature,
                "EXPECTED_SIGN": (
                    expected_sign
                ),
                "MODELS_EVALUATED": (
                    total_count
                ),
                "NEGATIVE_COUNT": int(
                    sign_counts.get(
                        "negative",
                        0,
                    )
                ),
                "POSITIVE_COUNT": int(
                    sign_counts.get(
                        "positive",
                        0,
                    )
                ),
                "ZERO_COUNT": int(
                    sign_counts.get(
                        "zero",
                        0,
                    )
                ),
                "EXPECTED_SIGN_MATCHES": (
                    matching_count
                ),
                "SIGN_CONSISTENCY_PERCENT": (
                    consistency_percent
                ),
                "MEAN_STANDARDIZED_COEFFICIENT": float(
                    subset[
                        "STANDARDIZED_COEFFICIENT"
                    ].mean()
                ),
                "MIN_STANDARDIZED_COEFFICIENT": float(
                    subset[
                        "STANDARDIZED_COEFFICIENT"
                    ].min()
                ),
                "MAX_STANDARDIZED_COEFFICIENT": float(
                    subset[
                        "STANDARDIZED_COEFFICIENT"
                    ].max()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    temporal_models = int(
        coefficients[
            [
                "SAMPLE",
                "TEMPORAL_ORIGIN",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    total_predictor_coefficients = len(
        predictor_coefficients
    )

    matching_coefficients = int(
        predictor_coefficients[
            "SIGN_MATCHES_EXPECTED"
        ].sum()
    )

    overall_consistency = float(
        100
        * matching_coefficients
        / total_predictor_coefficients
    )

    fully_stable_terms = int(
        summary[
            "SIGN_CONSISTENCY_PERCENT"
        ].eq(100.0).sum()
    )

    no_future_leakage = bool(
        (
            coefficients[
                "TEST_START_YEAR"
            ]
            > coefficients[
                "TEMPORAL_ORIGIN"
            ]
        ).all()
    )

    valid = (
        temporal_models == 5
        and len(summary) == 4
        and total_predictor_coefficients
        == temporal_models * 4
        and no_future_leakage
        and summary[
            "MODELS_EVALUATED"
        ].eq(
            temporal_models
        ).all()
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    NOW
                ),
                "TEMPORAL_MODELS": (
                    temporal_models
                ),
                "PREDICTORS_ANALYZED": (
                    len(summary)
                ),
                "PREDICTOR_COEFFICIENTS": (
                    total_predictor_coefficients
                ),
                "EXPECTED_SIGN_MATCHES": (
                    matching_coefficients
                ),
                "OVERALL_SIGN_CONSISTENCY_PERCENT": (
                    overall_consistency
                ),
                "FULLY_STABLE_TERMS": (
                    fully_stable_terms
                ),
                "NO_FUTURE_LEAKAGE": (
                    no_future_leakage
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
        "===== ESTABILIDAD DE "
        "COEFICIENTES dC/dt ====="
    )

    print()
    print(
        coefficients.loc[
            coefficients[
                "TERM"
            ].isin(FEATURES),
            [
                "SAMPLE",
                "TEMPORAL_ORIGIN",
                "BEST_ALPHA",
                "TERM",
                "STANDARDIZED_COEFFICIENT",
                "SIGN",
                "EXPECTED_SIGN",
                "SIGN_MATCHES_EXPECTED",
            ],
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== RESUMEN POR VARIABLE ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "MODELOS TEMPORALES:",
        temporal_models,
    )

    print(
        "COEFICIENTES ANALIZADOS:",
        total_predictor_coefficients,
    )

    print(
        "COINCIDENCIA GENERAL DE SIGNOS:",
        f"{overall_consistency:.1f}%",
    )

    print(
        "VARIABLES 100% ESTABLES:",
        fully_stable_terms,
        "de 4",
    )

    print(
        "FUGA TEMPORAL:",
        "NO"
        if no_future_leakage
        else "SÍ",
    )

    print(
        "VALIDACION_STATUS:",
        diagnostic.loc[
            0,
            "VALIDATION_STATUS",
        ],
    )

    print(
        "ARCHIVO:",
        COEFFICIENTS_FILE,
    )

    if not valid:
        raise ValueError(
            "La estabilidad de coeficientes "
            "requiere revisión."
        )


if __name__ == "__main__":
    main()