csfwm_6p = ["csf_wm",
            "trans_x",
            "trans_y",
            "trans_z",
            "rot_x",
            "rot_y",
            "rot_z"]

gs_csfwm_6p = ["global_signal",
               "csf_wm",
               "trans_x",
               "trans_y",
               "trans_z",
               "rot_x",
               "rot_y",
               "rot_z"]

csfwm_12p = ["csf_wm",
             "trans_x", "trans_x_derivative1",
             "trans_y", "trans_y_derivative1",
             "trans_z", "trans_z_derivative1",
             "rot_x", "rot_x_derivative1",
             "rot_y", "rot_y_derivative1",
             "rot_z", "rot_z_derivative1",]

gs_csfwm_12p = ["global_signal",
                "csf_wm",
                "trans_x", "trans_x_derivative1",
                "trans_y", "trans_y_derivative1",
                "trans_z", "trans_z_derivative1",
                "rot_x", "rot_x_derivative1",
                "rot_y", "rot_y_derivative1",
                "rot_z", "rot_z_derivative1",]

csfwm_24p = ["csf_wm",
             "trans_x", "trans_x_derivative1", "trans_x_derivative1_power2", "trans_x_power2",
             "trans_y", "trans_y_derivative1", "trans_y_power2", "trans_y_derivative1_power2",
             "trans_z", "trans_z_derivative1", "trans_z_derivative1_power2", "trans_z_power2",
             "rot_x", "rot_x_derivative1", "rot_x_derivative1_power2", "rot_x_power2",
             "rot_y", "rot_y_derivative1", "rot_y_derivative1_power2", "rot_y_power2",
             "rot_z", "rot_z_derivative1", "rot_z_derivative1_power2", "rot_z_power2"]

gs_csfwm_24p = ["global_signal",
                "csf_wm",
                "trans_x", "trans_x_derivative1", "trans_x_derivative1_power2", "trans_x_power2",
                "trans_y", "trans_y_derivative1", "trans_y_power2", "trans_y_derivative1_power2",
                "trans_z", "trans_z_derivative1", "trans_z_derivative1_power2", "trans_z_power2",
                "rot_x", "rot_x_derivative1", "rot_x_derivative1_power2", "rot_x_power2",
                "rot_y", "rot_y_derivative1", "rot_y_derivative1_power2", "rot_y_power2",
                "rot_z", "rot_z_derivative1", "rot_z_derivative1_power2", "rot_z_power2"]

denoising_strategies = {}
denoising_strategies["csfwm_6p"] = csfwm_6p
denoising_strategies["gs_csfwm_6p"] = gs_csfwm_6p
denoising_strategies["csfwm_12p"] = csfwm_12p
denoising_strategies["gs_csfwm_12p"] = gs_csfwm_12p
denoising_strategies["csfwm_24p"] = csfwm_24p
denoising_strategies["gs_csfwm_24p"] = gs_csfwm_24p