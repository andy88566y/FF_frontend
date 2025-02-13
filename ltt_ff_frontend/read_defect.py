# TODO: This file should be removed as soon as API support lrf details

import os
import re

from loguru import logger


CLASSTYPE_MAPPING = {
    'classified': {
        'lrf_to_model': {
            -1: -1,
            0: 1,
            1: 1, # 0 & 1 is defect, everything else is non-defect
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0,
            11: 0,
            12: 0,
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 0,
            18: 0,
            19: 0,
            20: 0,
            21: 0,
            22: 0,
            23: 0,
            24: 0,
            25: 0,
            26: 0,
            27: 0,
            28: 0,
            29: 0,
            30: 0,
            31: 0,
        },
        'model_to_lrf': {
            -1: -1, # unclassified remains as unclassfied
            0: 2,   # mapping model's prediction to classified.lrf non-defect label
            1: 1,   # mapping model's prediction to classified.lrf defect label
        }
    },
    'Classified': {
        'lrf_to_model': {
            -1: -1,
            0: 1,
            1: 1, # 0 & 1 is defect, everything else is non-defect
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0,
            11: 0,
            12: 0,
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 0,
            18: 0,
            19: 0,
            20: 0,
            21: 0,
            22: 0,
            23: 0,
            24: 0,
            25: 0,
            26: 0,
            27: 0,
            28: 0,
            29: 0,
            30: 0,
            31: 0,
        },
        'model_to_lrf': {
            -1: -1, # unclassified remains as unclassfied
            0: 2,   # mapping model's prediction to classified.lrf non-defect label
            1: 1,   # mapping model's prediction to classified.lrf defect label
        }
    },
    'ADD': {
        'lrf_to_model': {
            -1: -1,
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 1,
            9: 1,
            10: 0, # 10 is non-defect, everything else is defect
            11: 1,
            12: 1,
            13: 1,
            14: 1,
            15: 1,
            16: 1,
            17: 1,
            18: 1,
            19: 1,
            20: 1,
            21: 1,
            22: 1,
            23: 1,
            24: 1,
            25: 1,
            26: 1,
            27: 1,
            28: 1,
            29: 1,
            30: 1,
            31: 1,
        },
        'model_to_lrf': {
            -1: -1, # unclassified remains as unclassfied
            0: 10,   # mapping model's prediction to ADD.lrf non-defect label
            1: 1,   # mapping model's prediction to ADD.lrf defect label
        }
    },
    # TODO: for filtered lrf, mapping should follow the original lrf type instead of below
    'filtered': {
        'lrf_to_model': {
            -1: -1,
            1: 1,
            2: 0,
        },
        'model_to_lrf': {
            -1: -1, # unclassified remains as unclassfied
            0: 10,   # mapping model's prediction to ADD.lrf non-defect label
            1: 1,   # mapping model's prediction to ADD.lrf defect label
        }
    },
    # TODO: for base lrf, everything is unclassified
    'base': {
        'lrf_to_model': {
            -1: -1,
        },
        'model_to_lrf': {
            -1: -1, # unclassified remains as unclassfied
            0: 10,   # mapping model's prediction to ADD.lrf non-defect label
            1: 1,   # mapping model's prediction to ADD.lrf defect label
        }
    }
}


def get_lrf_type(lrf_path: str) -> str:

    logger.info(f'Getting lrf type from {lrf_path}')

    lrf_string = os.path.basename(lrf_path)
    if re.search(r".lrf$", lrf_string):
        lrf_string = re.sub(r".lrf$", "", lrf_string)

    if lrf_string.endswith(("_ADD", "_ADC", "_classified")):
        lrf_type = lrf_string.split('_')[-1]
        logger.success(f'{os.path.basename(lrf_path)} is a [{lrf_type}] lrf file.')
        return lrf_type
    elif re.search(r'_filtered_\d{6}$', lrf_string) is not None:
        logger.success(f'{os.path.basename(lrf_path)} is a [filtered] lrf file.')
        return 'filtered'
    else:
        logger.success(f'{os.path.basename(lrf_path)} is a [base] lrf file.')
        return 'base'


def lrf_parser(lrf_path: str) -> list[dict[str, str]]:
    logger.info(f"Reading LRF file from {lrf_path}")

    lrf_type = get_lrf_type(lrf_path)

    # Parse out the defect table at the of the lrf file
    defect_table_str = []
    bw_defect_table = False
    with open(lrf_path, encoding="utf-8") as f:
        for line in f:
            clean_line = line.replace("\n", "")

            if bw_defect_table:
                defect_table_str.append(clean_line)

            if clean_line == "[DefectList]":
                bw_defect_table = True

    # Remove "DefectDataColumn" and remove ; from "Comment;""
    column_names = defect_table_str[0].split(" ")[1:]
    column_names[-1] = column_names[-1][:-1]

    # Skip the first two header rows when iterating
    defects_info = []
    for defect_str in defect_table_str[2:]:
        defect_dat = list(filter(lambda x: x not in ["", ";"], defect_str.split(" ")))
        # Add blank comment if no comment found
        if len(defect_dat) == 18:
            defect_dat.append("")
        assert len(column_names) == len(defect_dat), (len(column_names), len(defect_dat))
        defects_info.append(dict(zip(column_names, defect_dat)))

    # Add 'isDefect' value according to ClassType
    for info in defects_info:
        info['isDefect'] = CLASSTYPE_MAPPING[lrf_type]["lrf_to_model"].get(int(info["ClassType"]), -1)

    logger.success(f"Successfully parsed {len(defects_info)} defects from lrf file.")

    return {int(defect["No"]): defect for defect in defects_info}
