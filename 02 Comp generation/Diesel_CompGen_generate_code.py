import os
import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel
import matplotlib.pyplot as plt
import time
from Bulk_prop_cal_gpu import (
    mixing_rule_density,
    mixing_rule_element_content_wt,
    mixing_rule_GC_MS_type_diesel,
    mixing_rule_distillation_cum_fraction_d86,
    mixing_rule_gasoline_PIONA,
    process_mol_data,
    mixing_rule_cetane_number,
    mixing_rule_freeze_point
)
from Mol_fraction_conversion import fraction_conversion
from typing import Tuple, Optional, List, Dict, Union

BASE_DIR = "./02 Comp generation"

FILE_PATHS = {
    'mol_index': os.path.join(BASE_DIR, "Fuel_mol_index.csv"),
    'mol_info': os.path.join(BASE_DIR, "Fuel_mol_infor_list.csv"),
    'checkpoint': os.path.join(BASE_DIR, "load_model/model_step_500000.pth"),
    'preprocess': os.path.join(BASE_DIR, "load_model"),
    'output': os.path.join(BASE_DIR, "generated_matrices")
}

MOL_COLUMNS = [
    'tb', 'vm_298K', 'C_number', 'H_number', 'S_number',
    'N_number', 'O_number', 'mw', 'aromatic_ring_number',
    'naphthenic_ring_number', 'thiophene_ring_number',
    'total_ring_number', 'gasoline_piona_type'
]

KEY_PROPERTIES = [
    'distillation_cum_fraction_d86_ibpv',
    'distillation_cum_fraction_d86_10v',
    'distillation_cum_fraction_d86_30v',
    'distillation_cum_fraction_d86_50v',
    'distillation_cum_fraction_d86_70v',
    'distillation_cum_fraction_d86_90v',
    'distillation_cum_fraction_d86_fbpv'
]

OPTIONAL_PROPS = [
    'density_288K',
    'density_293K',
    'freeze_point',
    'cetane_number',
    'RON',
    'reid_vapor_pressure',
    'C_weight_percentage',
    'H_weight_percentage',
    'S_weight_percentage',
    'N_weight_percentage',
    'gasoline_P_mass_fraction',
    'gasoline_I_mass_fraction',
    'gasoline_O_mass_fraction',
    'gasoline_N_mass_fraction',
    'gasoline_A_mass_fraction',
    'GCMS_group_P_weight_fraction',
    'GCMS_group_1N_weight_fraction',
    'GCMS_group_2N_weight_fraction',
    'GCMS_group_3N_weight_fraction',
    'GCMS_group_1A_weight_fraction',
    'GCMS_group_2A_weight_fraction',
    'GCMS_group_3A_weight_fraction'
]

MODEL_PARAMS = {
    'batch_size': 64,
    'target_size': 64,
    'mol_matrix_size': 64,
    'crop_size': 36,
    'num_timesteps': 1000,
    'scheduler_steps': 50
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_molecular_data() -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mol_index_df = pd.read_csv(FILE_PATHS['mol_index'], header=None)
    mol_index_tensor = torch.tensor(mol_index_df.values, dtype=torch.float32)

    mol_info_df = pd.read_csv(FILE_PATHS['mol_info'])
    mol_tensors = {}

    for column in MOL_COLUMNS:
        mol_tensors[column] = torch.tensor(mol_info_df[column].values, dtype=torch.float32)

    return mol_index_tensor, mol_tensors


def preprocess_input(properties_list: pd.Series, scaler: Optional[MinMaxScaler] = None) \
        -> Tuple[torch.Tensor, MinMaxScaler, int]:
    batch_data = []

    for properties in properties_list:
        key_data = np.array([properties.get(prop, 0.0) for prop in KEY_PROPERTIES]).reshape(1, -1)
        key_mask = np.ones_like(key_data)

        optional_data = []
        optional_mask = []

        for prop in OPTIONAL_PROPS:
            value = properties.get(prop, 0)

            if value != 0 and not pd.isna(value):
                optional_data.append(value)
                optional_mask.append(1)
            else:
                optional_data.append(0)
                optional_mask.append(0)

        optional_data = np.array(optional_data).reshape(1, -1)
        optional_mask = np.array(optional_mask).reshape(1, -1)

        combined_data = np.hstack([key_data, optional_data, key_mask, optional_mask])
        batch_data.append(combined_data)

    batch_data = np.vstack(batch_data)

    if scaler is None:
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(batch_data)
    else:
        x_scaled = scaler.transform(batch_data)

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    padding_size = MODEL_PARAMS['target_size'] - x_tensor.size(1)

    if padding_size > 0:
        padding = torch.zeros(x_tensor.size(0), padding_size, dtype=torch.float32)
        x_tensor = torch.cat((x_tensor, padding), dim=1)

    num_optional_props = len(OPTIONAL_PROPS)

    return x_tensor, scaler, num_optional_props


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * (torch.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).detach().to(torch.float32)


def load_trained_model() -> Tuple[UNet2DConditionModel, DDPMScheduler, DDPMPipeline]:
    betas = cosine_beta_schedule(MODEL_PARAMS['num_timesteps'])

    unet = UNet2DConditionModel(
        sample_size=MODEL_PARAMS['mol_matrix_size'],
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=MODEL_PARAMS['target_size']
    )

    checkpoint = torch.load(FILE_PATHS['checkpoint'], map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet = unet.to(device)
    unet.eval()

    scheduler = DDPMScheduler(
        num_train_timesteps=MODEL_PARAMS['num_timesteps'],
        beta_schedule="custom",
        trained_betas=betas.clone().detach().numpy())
    pipeline = DDPMPipeline(unet=unet, scheduler=scheduler)
    pipeline.to(device)

    return unet, scheduler, pipeline


def calculate_properties_from_matrix_batch(
        mol_matrix_batch: torch.Tensor,
        mol_index_tensor: torch.Tensor,
        mol_tensors: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    batch_size = mol_matrix_batch.size(0)
    mol_index_tensor = mol_index_tensor.unsqueeze(0).unsqueeze(0)
    mol_index_tensor = mol_index_tensor.repeat(batch_size, 1, 1, 1)

    x_mass = process_mol_data(mol_index_tensor, mol_matrix_batch, device=device)

    x_molar = fraction_conversion(
        mol_tensors['mw'],
        mol_tensors['vm_298K'],
        x_mass, 'mass', 'molar',
        device=device
    )

    x_volume = fraction_conversion(
        mol_tensors['mw'],
        mol_tensors['vm_298K'],
        x_molar, 'molar',
        'volume',
        device=device
    )

    properties = {}

    properties['density_293K'] = mixing_rule_density(
        mol_tensors['mw'],
        mol_tensors['vm_298K'],
        x_volume, T=293.15,
        device=device
    )

    C, H, S, N, O = mixing_rule_element_content_wt(
        mol_tensors['C_number'],
        mol_tensors['H_number'],
        mol_tensors['S_number'],
        mol_tensors['N_number'],
        mol_tensors['O_number'],
        mol_tensors['mw'],
        x_mass,
        device=device
    )

    properties['C_weight_percentage'] = C
    properties['H_weight_percentage'] = H
    properties['S_weight_percentage'] = S
    properties['N_weight_percentage'] = N

    (alkane_weight_fraction,
     one_ring_naphthene_weight_fraction,
     two_ring_naphthene_weight_fraction,
     three_ring_naphthene_weight_fraction,
     four_ring_naphthene_weight_fraction,
     five_ring_naphthene_weight_fraction,
     six_and_more_ring_naphthene_weight_fraction,
     one_ring_aromatic_weight_fraction,
     two_ring_aromatic_weight_fraction,
     three_ring_aromatic_weight_fraction,
     four_ring_aromatic_weight_fraction,
     five_ring_aromatic_weight_fraction,
     six_and_more_ring_aromatic_weight_fraction) = mixing_rule_GC_MS_type_diesel(
        mol_tensors['S_number'],
        mol_tensors['N_number'],
        mol_tensors['O_number'],
        mol_tensors['aromatic_ring_number'],
        mol_tensors['naphthenic_ring_number'],
        x_mass,
        device=device
    )

    properties['GCMS_group_P_weight_fraction'] = alkane_weight_fraction
    properties['GCMS_group_1N_weight_fraction'] = one_ring_naphthene_weight_fraction
    properties['GCMS_group_2N_weight_fraction'] = two_ring_naphthene_weight_fraction
    properties['GCMS_group_3N_weight_fraction'] = three_ring_naphthene_weight_fraction
    properties['GCMS_group_1A_weight_fraction'] = one_ring_aromatic_weight_fraction
    properties['GCMS_group_2A_weight_fraction'] = two_ring_aromatic_weight_fraction
    properties['GCMS_group_3A_weight_fraction'] = three_ring_aromatic_weight_fraction

    distillation_d86 = mixing_rule_distillation_cum_fraction_d86(
        x_volume,
        mol_tensors['tb'],
        device=device
    )

    properties['distillation_cum_fraction_d86_ibpv'] = distillation_d86[0]
    properties['distillation_cum_fraction_d86_10v'] = distillation_d86[2]
    properties['distillation_cum_fraction_d86_30v'] = distillation_d86[6]
    properties['distillation_cum_fraction_d86_50v'] = distillation_d86[10]
    properties['distillation_cum_fraction_d86_70v'] = distillation_d86[14]
    properties['distillation_cum_fraction_d86_90v'] = distillation_d86[18]
    properties['distillation_cum_fraction_d86_fbpv'] = distillation_d86[20]

    (gasoline_P_mass_fraction,
     gasoline_I_mass_fraction,
     gasoline_O_mass_fraction,
     gasoline_N_mass_fraction,
     gasoline_A_mass_fraction) = mixing_rule_gasoline_PIONA(
        x_mass,
        mol_tensors['gasoline_piona_type'],
        device=device
    )

    properties['gasoline_P_mass_fraction'] = gasoline_P_mass_fraction
    properties['gasoline_I_mass_fraction'] = gasoline_I_mass_fraction
    properties['gasoline_O_mass_fraction'] = gasoline_O_mass_fraction
    properties['gasoline_N_mass_fraction'] = gasoline_N_mass_fraction
    properties['gasoline_A_mass_fraction'] = gasoline_A_mass_fraction

    return properties


def generate_composition_from_properties(
        properties_list: Dict[str, float],
        mol_index_tensor: torch.Tensor,
        mol_tensors: Dict[str, torch.Tensor]
) -> np.ndarray:
    os.makedirs(FILE_PATHS['output'], exist_ok=True)

    batch_size = len(properties_list)

    for i, properties in enumerate(properties_list):
        missing_keys = set(KEY_PROPERTIES) - set(properties.keys())
        if missing_keys:
            print(f"Warning: Sample {i + 1} missing required properties: {missing_keys}")

        for prop in OPTIONAL_PROPS:
            if prop not in properties:
                properties[prop] = -1

    scaler_path = os.path.join(FILE_PATHS['preprocess'], "input_scaler_mask.pkl")
    y_max_path = os.path.join(FILE_PATHS['preprocess'], "y_max_mask.csv")

    loaded_scaler = joblib.load(scaler_path)
    y_max_df_loaded = pd.read_csv(y_max_path)
    y_max_loaded = y_max_df_loaded["y_max"].iloc[0]

    prop_series = pd.Series(properties)
    input_tensor, scale, num_optional_props = preprocess_input(properties_list, loaded_scaler)

    torch.manual_seed(42)
    initial_noise = torch.randn(batch_size, 1, MODEL_PARAMS['mol_matrix_size'], MODEL_PARAMS['mol_matrix_size'], device=device)

    print("Loading model...")
    unet, scheduler, pipeline = load_trained_model()

    start_time = time.time()

    print("Generating composition matrix...")
    with torch.no_grad():
        noise_input = initial_noise
        scheduler.set_timesteps(MODEL_PARAMS['scheduler_steps'])
        for t in scheduler.timesteps:
            noisy_residual = unet(noise_input, t, input_tensor.unsqueeze(1).to(device)).sample
            noise_input = scheduler.step(noisy_residual, t, noise_input).prev_sample

        cropped_image = noise_input[:, :, 14:50, 14:50]
        generated_images = cropped_image * y_max_loaded

    calculated_properties = calculate_properties_from_matrix_batch(
        generated_images, mol_index_tensor, mol_tensors
    )

    elapsed = time.time() - start_time
    print(f"Fine-tuning completed in {elapsed:.2f}s.")

    mol_matrices = []
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    properties_df = pd.DataFrame()

    properties_df['sample_id'] = [f"sample_{i + 1}" for i in range(batch_size)]

    for prop_name, prop_values in calculated_properties.items():
        properties_df[prop_name] = prop_values.cpu().numpy().flatten()

    properties_filename = f"calculated_properties_{timestamp}.csv"
    properties_path = os.path.join(FILE_PATHS['output'], properties_filename)
    properties_df.to_csv(properties_path, index=False)
    print(f"Saved calculated properties to: {properties_path}")

    for i in range(batch_size):
        mol_matrix = generated_images[i].squeeze().cpu().numpy()
        mol_matrices.append(mol_matrix)

        filename = f"generated_matrix_{timestamp}_{i + 1}.csv"
        output_path = os.path.join(FILE_PATHS['output'], filename)
        pd.DataFrame(mol_matrix).to_csv(output_path, index=False, header=False)
        print(f"Saved matrix {i + 1} to: {output_path}")

    return mol_matrix


if __name__ == "__main__":
    input_properties_list = [
        {
            'distillation_cum_fraction_d86_ibpv': 466.65,
            'distillation_cum_fraction_d86_10v': 520.75,
            'distillation_cum_fraction_d86_30v': 542.65,
            'distillation_cum_fraction_d86_50v': 560.85,
            'distillation_cum_fraction_d86_70v': 585.45,
            'distillation_cum_fraction_d86_90v': 624.55,
            'distillation_cum_fraction_d86_fbpv': 646.85,

            'density_293K': 0.8424,
            'C_weight_percentage': 85.08,
            'H_weight_percentage': 13.24,
            'S_weight_percentage': 1.351,
            'N_weight_percentage': 0.0077,
            'GCMS_group_P_weight_fraction': 0.472,
            'GCMS_group_1N_weight_fraction': 0.122,
            'GCMS_group_2N_weight_fraction': 0.086,
            'GCMS_group_3N_weight_fraction': 0.022,
            'GCMS_group_1A_weight_fraction': 0.185,
            'GCMS_group_2A_weight_fraction': 0.100,
            'GCMS_group_3A_weight_fraction': 0.013,
            'gasoline_P_mass_fraction': 0.067,
            'gasoline_I_mass_fraction': 0.202,
            'gasoline_O_mass_fraction': 0.202,
            'gasoline_N_mass_fraction': 0.23,
            'gasoline_A_mass_fraction': 0.298,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 484.65,
            'distillation_cum_fraction_d86_10v': 513.35,
            'distillation_cum_fraction_d86_30v': 526.05,
            'distillation_cum_fraction_d86_50v': 538.25,
            'distillation_cum_fraction_d86_70v': 553.45,
            'distillation_cum_fraction_d86_90v': 577.65,
            'distillation_cum_fraction_d86_fbpv': 594.35,

            'density_293K': 0.8202,
            'C_weight_percentage': 85.5,
            'H_weight_percentage': 13.92,
            'S_weight_percentage': 0.077,
            'N_weight_percentage': 0.0055,
            'GCMS_group_P_weight_fraction': 0.522,
            'GCMS_group_1N_weight_fraction': 0.145,
            'GCMS_group_2N_weight_fraction': 0.142,
            'GCMS_group_3N_weight_fraction': 0.047,
            'GCMS_group_1A_weight_fraction': 0.086,
            'GCMS_group_2A_weight_fraction': 0.056,
            'GCMS_group_3A_weight_fraction': 0.002,
            'gasoline_P_mass_fraction': 0.074,
            'gasoline_I_mass_fraction': 0.223,
            'gasoline_O_mass_fraction': 0.223,
            'gasoline_N_mass_fraction': 0.334,
            'gasoline_A_mass_fraction': 0.144,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 455.05,
            'distillation_cum_fraction_d86_10v': 505.35,
            'distillation_cum_fraction_d86_30v': 528.95,
            'distillation_cum_fraction_d86_50v': 545.45,
            'distillation_cum_fraction_d86_70v': 564.75,
            'distillation_cum_fraction_d86_90v': 596.75,
            'distillation_cum_fraction_d86_fbpv': 625.75,

            'density_293K': 0.8318,
            'C_weight_percentage': 85.52,
            'H_weight_percentage': 13.69,
            'S_weight_percentage': 0.788,
            'N_weight_percentage': 0.0086,
            'GCMS_group_P_weight_fraction': 0.465,
            'GCMS_group_1N_weight_fraction': 0.153,
            'GCMS_group_2N_weight_fraction': 0.084,
            'GCMS_group_3N_weight_fraction': 0.002,
            'GCMS_group_1A_weight_fraction': 0.188,
            'GCMS_group_2A_weight_fraction': 0.082,
            'GCMS_group_3A_weight_fraction': 0.008,
            'gasoline_P_mass_fraction': 0.066,
            'gasoline_I_mass_fraction': 0.199,
            'gasoline_O_mass_fraction': 0.199,
            'gasoline_N_mass_fraction': 0.239,
            'gasoline_A_mass_fraction': 0.278,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 467.35,
            'distillation_cum_fraction_d86_10v': 508.35,
            'distillation_cum_fraction_d86_30v': 525.75,
            'distillation_cum_fraction_d86_50v': 536.95,
            'distillation_cum_fraction_d86_70v': 548.35,
            'distillation_cum_fraction_d86_90v': 565.85,
            'distillation_cum_fraction_d86_fbpv': 580.15,

            'density_293K': 0.8146,
            'C_weight_percentage': 85.7,
            'H_weight_percentage': 13.93,
            'S_weight_percentage': 0.03,
            'N_weight_percentage': 0.0018,
            'GCMS_group_P_weight_fraction': 0.546,
            'GCMS_group_1N_weight_fraction': 0.124,
            'GCMS_group_2N_weight_fraction': 0.140,
            'GCMS_group_3N_weight_fraction': 0.031,
            'GCMS_group_1A_weight_fraction': 0.090,
            'GCMS_group_2A_weight_fraction': 0.068,
            'GCMS_group_3A_weight_fraction': 0.001,
            'gasoline_P_mass_fraction': 0.078,
            'gasoline_I_mass_fraction': 0.234,
            'gasoline_O_mass_fraction': 0.234,
            'gasoline_N_mass_fraction': 0.295,
            'gasoline_A_mass_fraction': 0.159,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 478.35,
            'distillation_cum_fraction_d86_10v': 500.45,
            'distillation_cum_fraction_d86_30v': 516.15,
            'distillation_cum_fraction_d86_50v': 529.75,
            'distillation_cum_fraction_d86_70v': 542.85,
            'distillation_cum_fraction_d86_90v': 560.15,
            'distillation_cum_fraction_d86_fbpv': 572.45,

            'density_293K': 0.8791,
            'C_weight_percentage': 86.22,
            'H_weight_percentage': 12.88,
            'S_weight_percentage': 0.062,
            'N_weight_percentage': 0.0057,
            'GCMS_group_P_weight_fraction': 0.116,
            'GCMS_group_1N_weight_fraction': 0.003,
            'GCMS_group_2N_weight_fraction': 0.447,
            'GCMS_group_3N_weight_fraction': 0.201,
            'GCMS_group_1A_weight_fraction': 0.177,
            'GCMS_group_2A_weight_fraction': 0.055,
            'GCMS_group_3A_weight_fraction': 0.001,
            'gasoline_P_mass_fraction': 0.016,
            'gasoline_I_mass_fraction': 0.050,
            'gasoline_O_mass_fraction': 0.050,
            'gasoline_N_mass_fraction': 0.651,
            'gasoline_A_mass_fraction': 0.233,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 477.15,
            'distillation_cum_fraction_d86_10v': 530.85,
            'distillation_cum_fraction_d86_30v': 556.95,
            'distillation_cum_fraction_d86_50v': 583.75,
            'distillation_cum_fraction_d86_70v': 611.05,
            'distillation_cum_fraction_d86_90v': 636.05,
            'distillation_cum_fraction_d86_fbpv': 650.15,

            'density_293K': 0.903,
            'C_weight_percentage': 85.11,
            'H_weight_percentage': 11.49,
            'S_weight_percentage': 0.091,
            'N_weight_percentage': 0.077,
            'GCMS_group_P_weight_fraction': 0.245,
            'GCMS_group_1N_weight_fraction': 0.116,
            'GCMS_group_2N_weight_fraction': 0.079,
            'GCMS_group_3N_weight_fraction': 0.050,
            'GCMS_group_1A_weight_fraction': 0.244,
            'GCMS_group_2A_weight_fraction': 0.194,
            'GCMS_group_3A_weight_fraction': 0.072,
            'gasoline_P_mass_fraction': 0.035,
            'gasoline_I_mass_fraction': 0.105,
            'gasoline_O_mass_fraction': 0.105,
            'gasoline_N_mass_fraction': 0.245,
            'gasoline_A_mass_fraction': 0.510,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 444.95,
            'distillation_cum_fraction_d86_10v': 518.65,
            'distillation_cum_fraction_d86_30v': 539.95,
            'distillation_cum_fraction_d86_50v': 555.65,
            'distillation_cum_fraction_d86_70v': 572.25,
            'distillation_cum_fraction_d86_90v': 593.95,
            'distillation_cum_fraction_d86_fbpv': 610.45,

            'density_293K': 0.8245,
            'C_weight_percentage': 85.5,
            'H_weight_percentage': 13.92,
            'S_weight_percentage': 0.089,
            'N_weight_percentage': 0.0832,
            'GCMS_group_P_weight_fraction': 0.395,
            'GCMS_group_1N_weight_fraction': 0.262,
            'GCMS_group_2N_weight_fraction': 0.091,
            'GCMS_group_3N_weight_fraction': 0.032,
            'GCMS_group_1A_weight_fraction': 0.138,
            'GCMS_group_2A_weight_fraction': 0.074,
            'GCMS_group_3A_weight_fraction': 0.008,
            'gasoline_P_mass_fraction': 0.056,
            'gasoline_I_mass_fraction': 0.169,
            'gasoline_O_mass_fraction': 0.169,
            'gasoline_N_mass_fraction': 0.385,
            'gasoline_A_mass_fraction': 0.220,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 459.05,
            'distillation_cum_fraction_d86_10v': 509.85,
            'distillation_cum_fraction_d86_30v': 533.85,
            'distillation_cum_fraction_d86_50v': 561.95,
            'distillation_cum_fraction_d86_70v': 590.35,
            'distillation_cum_fraction_d86_90v': 624.55,
            'distillation_cum_fraction_d86_fbpv': 646.85,

            'density_293K': 0.886,
            'C_weight_percentage': 87.62,
            'H_weight_percentage': 11.61,
            'S_weight_percentage': 0.445,
            'N_weight_percentage': 0.0567,
            'GCMS_group_P_weight_fraction': 0.297,
            'GCMS_group_1N_weight_fraction': 0.096,
            'GCMS_group_2N_weight_fraction': 0.061,
            'GCMS_group_3N_weight_fraction': 0.021,
            'GCMS_group_1A_weight_fraction': 0.204,
            'GCMS_group_2A_weight_fraction': 0.274,
            'GCMS_group_3A_weight_fraction': 0.047,
            'gasoline_P_mass_fraction': 0.042,
            'gasoline_I_mass_fraction': 0.127,
            'gasoline_O_mass_fraction': 0.127,
            'gasoline_N_mass_fraction': 0.178,
            'gasoline_A_mass_fraction': 0.525,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 442.35,
            'distillation_cum_fraction_d86_10v': 501.65,
            'distillation_cum_fraction_d86_30v': 519.45,
            'distillation_cum_fraction_d86_50v': 544.15,
            'distillation_cum_fraction_d86_70v': 565.85,
            'distillation_cum_fraction_d86_90v': 579.75,
            'distillation_cum_fraction_d86_fbpv': 602.55,

            'density_293K': 0.906,
            'C_weight_percentage': 89.07,
            'H_weight_percentage': 10.86,
            'S_weight_percentage': 0.0559,
            'N_weight_percentage': 0.0165,
            'GCMS_group_P_weight_fraction': 0.224,
            'GCMS_group_1N_weight_fraction': 0.041,
            'GCMS_group_2N_weight_fraction': 0.034,
            'GCMS_group_3N_weight_fraction': 0.014,
            'GCMS_group_1A_weight_fraction': 0.211,
            'GCMS_group_2A_weight_fraction': 0.442,
            'GCMS_group_3A_weight_fraction': 0.034,
            'gasoline_P_mass_fraction': 0.032,
            'gasoline_I_mass_fraction': 0.096,
            'gasoline_O_mass_fraction': 0.096,
            'gasoline_N_mass_fraction': 0.089,
            'gasoline_A_mass_fraction': 0.687,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 460.75,
            'distillation_cum_fraction_d86_10v': 482.75,
            'distillation_cum_fraction_d86_30v': 521.05,
            'distillation_cum_fraction_d86_50v': 544.95,
            'distillation_cum_fraction_d86_70v': 572.65,
            'distillation_cum_fraction_d86_90v': 611.75,
            'distillation_cum_fraction_d86_fbpv': 672.75,

            'density_293K': 0.8183,
            'C_weight_percentage': 85.86,
            'H_weight_percentage': 14.14,
            'S_weight_percentage': 0.0003,
            'N_weight_percentage': 0,
            'GCMS_group_P_weight_fraction': 0.515,
            'GCMS_group_1N_weight_fraction': 0.185,
            'GCMS_group_2N_weight_fraction': 0.130,
            'GCMS_group_3N_weight_fraction': 0.045,
            'GCMS_group_1A_weight_fraction': 0.135,
            'GCMS_group_2A_weight_fraction': 0.005,
            'GCMS_group_3A_weight_fraction': 0.001,
            'gasoline_P_mass_fraction': 0.073,
            'gasoline_I_mass_fraction': 0.221,
            'gasoline_O_mass_fraction': 0.221,
            'gasoline_N_mass_fraction': 0.36,
            'gasoline_A_mass_fraction': 0.141,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 465.65,
            'distillation_cum_fraction_d86_10v': 490.75,
            'distillation_cum_fraction_d86_30v': 507.45,
            'distillation_cum_fraction_d86_50v': 530.65,
            'distillation_cum_fraction_d86_70v': 556.35,
            'distillation_cum_fraction_d86_90v': 582.95,
            'distillation_cum_fraction_d86_fbpv': 597.55,

            'density_293K': 0.931,
            'C_weight_percentage': 85.90,
            'H_weight_percentage': 13.19,
            'S_weight_percentage': 0.03456,
            'N_weight_percentage': 0.00412,
            'GCMS_group_P_weight_fraction': 0.17,
            'GCMS_group_1N_weight_fraction': 0.097,
            'GCMS_group_2N_weight_fraction': 0.031,
            'GCMS_group_3N_weight_fraction': 0.015,
            'GCMS_group_1A_weight_fraction': 0.258,
            'GCMS_group_2A_weight_fraction': 0.371,
            'GCMS_group_3A_weight_fraction': 0.058,
            'gasoline_P_mass_fraction': 0.024,
            'gasoline_I_mass_fraction': 0.072,
            'gasoline_O_mass_fraction': 0.072,
            'gasoline_N_mass_fraction': 0.143,
            'gasoline_A_mass_fraction': 0.687,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 447.45,
            'distillation_cum_fraction_d86_10v': 499.85,
            'distillation_cum_fraction_d86_30v': 533.89,
            'distillation_cum_fraction_d86_50v': 561.35,
            'distillation_cum_fraction_d86_70v': 584.35,
            'distillation_cum_fraction_d86_90v': 619.85,
            'distillation_cum_fraction_d86_fbpv': 638.12,

            'density_293K': 0.888,
            'C_weight_percentage': 85.52,
            'H_weight_percentage': 13.39,
            'S_weight_percentage': 1.08,
            'N_weight_percentage': 0.00831,
            'GCMS_group_P_weight_fraction': 0.227,
            'GCMS_group_1N_weight_fraction': 0.147,
            'GCMS_group_2N_weight_fraction': 0.144,
            'GCMS_group_3N_weight_fraction': 0.045,
            'GCMS_group_1A_weight_fraction': 0.241,
            'GCMS_group_2A_weight_fraction': 0.173,
            'GCMS_group_3A_weight_fraction': 0.023,
            'gasoline_P_mass_fraction': 0.033,
            'gasoline_I_mass_fraction': 0.097,
            'gasoline_O_mass_fraction': 0.097,
            'gasoline_N_mass_fraction': 0.336,
            'gasoline_A_mass_fraction': 0.437,
        }
    ]

    print("Loading molecular data...")
    mol_index_tensor, mol_tensors = load_molecular_data()

    print("Generating composition matrix...")
    pred_matrix = generate_composition_from_properties(
        properties_list=input_properties_list,
        mol_index_tensor=mol_index_tensor,
        mol_tensors=mol_tensors
    )

    print("Generation complete! Matrix dimensions:", pred_matrix.shape)