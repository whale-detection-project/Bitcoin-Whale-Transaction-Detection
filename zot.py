import pandas as pd

def check_wallet_address_overlap(old_wallet_path, top_1000_path):
    """
    old_wallet.csv 파일의 'Address' 컬럼과 top_1000.csv 파일의 'address' 컬럼에
    있는 지갑 주소들 간에 겹치는 부분이 있는지 확인하고, 겹치는 주소를 출력합니다.

    Args:
        old_wallet_path (str): old_wallet.csv 파일의 경로.
        top_1000_path (str): top_1000.csv 파일의 경로.
    """
    try:
        # old_wallet.csv 파일 불러오기 (컬럼명 'Address' 지정)
        df_old_wallet = pd.read_csv(old_wallet_path)
        # 'Address' 컬럼을 지갑 주소로 사용
        old_wallet_addresses = set(df_old_wallet['Address'].astype(str).str.strip().tolist())

        # top_1000.csv 파일 불러오기 (컬럼명 'address' 지정)
        df_top_1000 = pd.read_csv(top_1000_path)
        # 'address' 컬럼을 지갑 주소로 사용
        top_1000_addresses = set(df_top_1000['address'].astype(str).str.strip().tolist())

        # 겹치는 주소 찾기
        overlapping_addresses = old_wallet_addresses.intersection(top_1000_addresses)

        if overlapping_addresses:
            print("겹치는 지갑 주소가 있습니다:")
            for address in overlapping_addresses:
                print(address)
            print(f"총 겹치는 주소 개수: {len(overlapping_addresses)}개")
        else:
            print("겹치는 지갑 주소가 없습니다.")

    except FileNotFoundError:
        print("파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    except KeyError as e:
        print(f"컬럼을 찾을 수 없습니다: {e}. 파일의 컬럼명을 확인해주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

# 파일 경로 지정 (실제 파일 경로로 변경해주세요)
old_wallet_file = 'old_wallet.csv'
top_1000_file = 'top_1000_labeled_wallets.csv'

# 스크립트 실행
check_wallet_address_overlap(old_wallet_file, top_1000_file)