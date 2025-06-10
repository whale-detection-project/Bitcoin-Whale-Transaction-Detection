def process_transaction(tx):
    """거래 데이터를 처리하여 특성을 추출합니다."""
    total_input_value = sum(i.get("prev_out", {}).get("value", 0) for i in tx.get("inputs", [])) / 1e8

    input_count = len(tx.get("inputs", []))
    output_list = tx.get("out", [])
    output_count = len(output_list)
    output_values = [o.get("value", 0) for o in output_list]
    total_output_value = sum(output_values)
    max_output = max(output_values, default=0)

    fee = (total_input_value * 1e8 - total_output_value) if total_output_value else 0
    fee_per_max_ratio = fee / max_output if max_output else 0
    max_output_ratio = max_output / total_output_value if total_output_value else 0

    max_input = max([i.get("prev_out", {}).get("value", 0) for i in tx.get("inputs", [])], default=0)
    max_input_ratio = max_input / (total_input_value * 1e8) if total_input_value > 0 else 0

    return {
        "input_count": input_count,
        "output_count": output_count,
        "max_output_ratio": max_output_ratio,
        "fee_per_max_ratio": fee_per_max_ratio,
        "max_input_ratio": max_input_ratio,
        "total_input_value": total_input_value,
    }