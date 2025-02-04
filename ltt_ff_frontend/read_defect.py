from loguru import logger


def read_defects(lrf_path: str) -> dict[int, dict[str, str]]:
    logger.info(f"Reading LRF file from {lrf_path}")

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

    # Remove "DefectDataColumn" and "Comment;""
    column_names = defect_table_str[0].split(" ")[1:-1]

    logger.debug(f"lrf defect columns: {column_names}")

    # Skip the first two header rows when iterating
    defects_info = []
    for defect_str in defect_table_str[2:]:
        defect_dat = list(filter(lambda x: x not in ["", ";"], defect_str.split(" ")))
        assert len(column_names) == len(defect_dat), (len(column_names), len(defect_dat))
        defects_info.append(dict(zip(column_names, defect_dat)))

    logger.success(f"Successfully parsed {len(defects_info)} defects from lrf file.")

    return {int(defect["No"]): defect for defect in defects_info}
 
