## Dataset Description

This is a summary of all the csv fields you may find inside datasets of this repository.

- `name_i`: The name of the i-th person involved in the event (e.g., `name_1`, `name_2`, `name_3`).
- `date_i`: The date associated with the corresponding person’s event in `YYYY-MM-DD` format (e.g., `date_1`, `date_2`, `date_3`).
- `duration_str_i`: The duration of the i-th person’s event as expressed in the context (e.g., `4 days`, `2 weeks`).
- `duration_type`: The unit of time used to express the duration in the context (e.g., `days`, `weeks`).
- `period_length_i`: Average period length expressed in days.
- `period_str_i`: The period or frequency as expressed in the context (e.g., `every 5 days`, `five times a week`).
- `period_type`: The unit of the period/frequency used in the context (e.g., `days`, `weeks`).
- `time_i`: The time associated with the i-th person’s event in `HH:MM:SS` format (e.g., `07:30:00`).
- `date_change`: A boolean indicating whether the calculation crosses into a different day (`True` or `False`).
- `context`: The full context including all information as passed to the LLM.
- `alternatives`: A list of all possible names that could be the answer.
- `correct_answer`: The correct continuation to the context.
- `correct_date`: The date of the correct continuation, in `YYYY-MM-DD` format.
- `correct_date_str`: The date of the correct continuation as it appears in the context (e.g., `9th of January`).
alternatives: A list of all possible names involved in the event in Python list format (e.g., ['Ryan', 'Neil', 'Matt']).
- `correct_season`: The season (e.g., `spring`, `summer`, `fall`, `winter`) associated with the correct answer.
- `correct_season_label`: A numeric label representing the season (e.g., 1 for spring, 2 for summer, etc.).
- `correct_temperature`: A categorical label indicating temperature context (`cold`, `warm`) associated with the correct answer.
- `correct_temperature_label`: A numeric label representing the temperature context (0 for cold, 1 for warm).
- `correct_date_expr`: The date of the correct continuation as expressed in the context (e.g., `the 5th of February`).
- `correct_end_date`: The end date of the correct continuation’s event in `YYYY-MM-DD` format.
- `correct_duration`: The duration of the correct continuation’s event in a standardized format (e.g., `6 days`).
- `correct_duration_str`: The duration of the correct continuation’s event as expressed in the context (e.g., `6 days`, `2 weeks`).
- `correct_duration_length`: Length of the correct duration in days.
- `correct_month`: The month in which the correct event occurs (e.g., `February`).
- `correct_time`: The time of the correct continuation in `HH:MM:SS` format.
- `correct_time_expr`: The time of the correct continuation as expressed in the context (e.g., `7:30`).
- `correct_time_diff`: The difference in minutes between the current reference time and the correct time.
- `correct_phase`: The period of the day associated with the correct answer (e.g., `night`, `morning`, `afternoon`, `evening`).
- `correct_phase_label`: A numeric label representing the period of the day (e.g., 0 for night, 1 for morning, etc.).
- `time_idx_start`: The starting character index of the correct answer’s time expression in the context.
- `time_idx_end`: The ending character index of the correct answer’s time expression in the context.

