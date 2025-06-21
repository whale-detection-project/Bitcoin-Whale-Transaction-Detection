def process_transaction(tx):
    """거래 데이터를 처리하여 특성과 최대 입출력 주소를 추출합니다."""
    inputs = tx.get("inputs", [])
    outputs = tx.get("out", [])

    # 총 입력 금액 (BTC 기준)
    total_input_value = sum(i.get("prev_out", {}).get("value", 0) for i in inputs) / 1e8

    # 입출력 개수
    input_count = len(inputs)
    output_count = len(outputs)

    # output 관련 금액 계산
    output_values = [o.get("value", 0) for o in outputs]
    total_output_value = sum(output_values)
    max_output = max(output_values, default=0)

    # 수수료 및 비율 계산
    fee = (total_input_value * 1e8 - total_output_value) if total_output_value else 0
    fee_per_max_ratio = fee / max_output if max_output else 0
    max_output_ratio = max_output / total_output_value if total_output_value else 0

    # max input 계산 + 주소 추출
    max_input_value = 0
    max_input_address = None
    for i in inputs:
        val = i.get("prev_out", {}).get("value", 0)
        if val > max_input_value:
            max_input_value = val
            max_input_address = i.get("prev_out", {}).get("addr")

    max_input_ratio = max_input_value / (total_input_value * 1e8) if total_input_value > 0 else 0

    # max output 계산 + 주소 추출
    max_output_value = 0
    max_output_address = None
    for o in outputs:
        val = o.get("value", 0)
        if val > max_output_value:
            max_output_value = val
            max_output_address = o.get("addr")

    return {
        "input_count": input_count,
        "output_count": output_count,
        "max_output_ratio": max_output_ratio,
        "fee_per_max_ratio": fee_per_max_ratio,
        "max_input_ratio": max_input_ratio,
        "total_input_value": total_input_value,
        "max_input_address": max_input_address,
        "max_output_address": max_output_address
    }
