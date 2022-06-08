import os

casf_types_files = ['/data/localhost/not-backed-up/durant/casf_2016_pymol_redocked.types',
'/data/localhost/not-backed-up/durant/casf_2016_pymol_crossdocked_most_similiar_structure.types',
'/data/localhost/not-backed-up/durant/casf_2016_pymol_crossdocked_least_similiar_structure.types',
'/data/localhost/not-backed-up/durant/casf_2016_pymol.types']


models = [
'crystal_regression/PointBAP/crystal_regression/',
'crystal_regression/PointBAP/crystal_regression_seq_100/',
'crystal_regression/PointBAP/crystal_regression_seq_90/',
'crystal_regression/PointBAP/crystal_regression_seq_30/',
'crystal_regression/PointBAP/crystal_regression_tan_100/',
'crystal_regression/PointBAP/crystal_regression_tan_90/',
'crystal_regression/PointBAP/crystal_regression_tan_30/',
'crystal_regression/PointBAP/crystal_regression_both_100/',
'crystal_regression/PointBAP/crystal_regression_both_90/',
'crystal_regression/PointBAP/crystal_regression_both_30/',
'crystal_regression/PointBAP/crystal_regression_both_30_size_control/',
'docked_regression/PointBAP/docked_regression/',
'docked_regression/PointBAP/docked_regression_seq_100/',
'docked_regression/PointBAP/docked_regression_seq_90/',
'docked_regression/PointBAP/docked_regression_seq_30/',
'docked_regression/PointBAP/docked_regression_tan_100/',
'docked_regression/PointBAP/docked_regression_tan_90/',
'docked_regression/PointBAP/docked_regression_tan_30/',
'docked_regression/PointBAP/docked_regression_both_100/',
'docked_regression/PointBAP/docked_regression_both_90/',
'docked_regression/PointBAP/docked_regression_both_30/',
'docked_regression/PointBAP/docked_regression_both_30_size_control/',

]


for c in casf_types_files:
	for m in models:
		os.system(f"python point_vs/inference.py {m} {c} /data/localhost/not-backed-up/durant/pdbbind_2020_general_parquets  --wandb_project PointBAP --wandb_run {m.split('/')[-2]} ")