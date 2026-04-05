from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    domain: str
    title: str
    priority: str
    owner: str
    depends_on: tuple[str, ...] = ()
    figure: str | None = None


BENCHMARK_EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("B00", "benchmark", "Null calibration", "P0", "Methods + Stats", figure="Fig 3A"),
    ExperimentSpec("B01", "benchmark", "Linear stationary fixed lag", "P1", "Methods", ("B00",), "Fig 3B"),
    ExperimentSpec("B02", "benchmark", "Linear with varying lag", "P1", "Methods", ("B01",), "Fig 4A"),
    ExperimentSpec("B03", "benchmark", "Nonlinear polynomial", "P1", "Methods", ("B01",), "Fig 4B"),
    ExperimentSpec("B04", "benchmark", "Nonlinear with varying lag", "P1", "Methods", ("B02", "B03"), "Fig 4C"),
    ExperimentSpec("B05", "benchmark", "Nonstationary drift", "P1", "Methods", ("B01",), "Fig 4D"),
    ExperimentSpec("B06", "benchmark", "Nonstationary with varying lag", "P2", "Methods", ("B02", "B05"), "Fig S1"),
    ExperimentSpec("B07", "benchmark", "Time-varying edges", "P1", "Methods", ("B01",), "Fig 5A"),
    ExperimentSpec("B08", "benchmark", "Common-driver and indirect-path graphs", "P1", "Methods + Stats", ("B01",), "Fig 5B"),
    ExperimentSpec("B09", "benchmark", "Short-record regime", "P1", "Methods", ("B01", "B02", "B03", "B04"), "Fig 5C"),
    ExperimentSpec("B10", "benchmark", "Noise robustness", "P1", "Methods", ("B01", "B02", "B03", "B04", "B05"), "Fig 5D"),
    ExperimentSpec("B11", "benchmark", "High-dimensional sparse graphs", "P2", "Methods", ("B01", "B09"), "Fig 6A"),
    ExperimentSpec("B12", "benchmark", "Runtime and scaling", "P2", "Methods", ("B01", "B11"), "Fig 6B"),
    ExperimentSpec("B13", "benchmark", "Architecture ablation", "P2", "Methods", ("B03", "B04", "B07"), "Fig S2"),
    ExperimentSpec("B14", "benchmark", "Significance-test ablation", "P3", "Stats", ("B00",), "Fig S3"),
)


APPLICATION_EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("E00", "application", "Preprocessing pilot", "P0", "Data", figure="Fig 2A"),
    ExperimentSpec("E01", "application", "Full ROI-ready cohort", "P1", "Data + Methods", ("E00",), "Table 1"),
    ExperimentSpec("E02", "application", "Rest vs execution", "P1", "Data + Methods", ("E01",), "Fig 7A"),
    ExperimentSpec("E03", "application", "Rest vs imagery", "P1", "Data + Methods", ("E01",), "Fig 7B"),
    ExperimentSpec("E04", "application", "Execution vs imagery", "P1", "Data + Methods", ("E01",), "Fig 7C"),
    ExperimentSpec("E05", "application", "Left vs right hand", "P1", "Data + Methods", ("E01",), "Fig 8A"),
    ExperimentSpec("E06", "application", "Cross-subject reproducibility", "P1", "Stats", ("E02", "E03", "E04", "E05"), "Fig 8B"),
    ExperimentSpec("E07", "application", "Graph-level summaries", "P1", "Stats", ("E02", "E03", "E04", "E05"), "Table 2"),
    ExperimentSpec("E08", "application", "Biological plausibility review", "P1", "Writing", ("E02", "E03", "E04", "E05", "E06", "E07"), "Discussion"),
    ExperimentSpec("E09", "application", "Preprocessing sensitivity", "P2", "Data + Methods", ("E02", "E03", "E04", "E05"), "Fig S4"),
    ExperimentSpec("E10", "application", "Epoch-length sensitivity", "P2", "Data + Methods", ("E02", "E03", "E04", "E05"), "Fig S5"),
    ExperimentSpec("E11", "application", "Channel-set sensitivity", "P2", "Data + Methods", ("E02", "E03", "E04", "E05"), "Fig S6"),
    ExperimentSpec("E12", "application", "Band-specific analysis", "P2", "Data + Methods", ("E02", "E03", "E04", "E05"), "Fig S7"),
    ExperimentSpec("E13", "application", "Label-shuffle or temporal-destroy control", "P1", "Stats", ("E02", "E03", "E04", "E05"), "Fig S8"),
    ExperimentSpec("E14", "application", "Subject-subsampling stability", "P2", "Stats", ("E02", "E03", "E04", "E05", "E06", "E07"), "Fig S9"),
)


ALL_EXPERIMENTS = BENCHMARK_EXPERIMENTS + APPLICATION_EXPERIMENTS


def get_experiment(experiment_id: str) -> ExperimentSpec:
    for spec in ALL_EXPERIMENTS:
        if spec.id == experiment_id:
            return spec
    raise KeyError(f"Unknown experiment id: {experiment_id}")
