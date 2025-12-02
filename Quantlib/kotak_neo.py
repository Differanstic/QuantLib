import neo_api_client

class KotakNeo:
    def __init__(self, consumer_key, mobile_number, ucc, mpin):
        self.client = neo_api_client.NeoAPI(environment="PROD", consumer_key=consumer_key)
        self.mobile_number = mobile_number
        self.ucc = ucc
        self.mpin = mpin
        self.totp_secret = input('Enter TOTP: ')
        self._login()

    def _login(self):
        self.client.totp_login(mobile_number=self.mobile_number, ucc=self.ucc, totp=self.totp_secret)
        dict = self.client.totp_validate(mpin=self.mpin)
        if dict['data']['status'] == 'success':
            print("Successfully logged in to Kotak Neo.")
        else :
            print('Error in Login')

    def place_order(self, exchange_segment, product, price, order_type, quantity, validity, trading_symbol,
                    transaction_type, amo="NO", disclosed_quantity="0", market_protection="0", pf="N",
                    trigger_price="0", tag=None, scrip_token=None, square_off_type=None,
                    stop_loss_type=None, stop_loss_value=None, square_off_value=None,
                    last_traded_price=None, trailing_stop_loss=None, trailing_sl_value=None):
        return self.client.place_order(
            exchange_segment=exchange_segment, product=product, price=price, order_type=order_type,
            quantity=quantity, validity=validity, trading_symbol=trading_symbol,
            transaction_type=transaction_type, amo=amo, disclosed_quantity=disclosed_quantity,
            market_protection=market_protection, pf=pf, trigger_price=trigger_price, tag=tag,
            scrip_token=scrip_token, square_off_type=square_off_type, stop_loss_type=stop_loss_type,
            stop_loss_value=stop_loss_value, square_off_value=square_off_value,
            last_traded_price=last_traded_price, trailing_stop_loss=trailing_stop_loss,
            trailing_sl_value=trailing_sl_value
        )

    def modify_order(self, order_id, price, quantity, disclosed_quantity="0", trigger_price="0",
                     validity="DAY", order_type=''):
        return self.client.modify_order(
            order_id=order_id, price=price, quantity=quantity,
            disclosed_quantity=disclosed_quantity, trigger_price=trigger_price,
            validity=validity, order_type=order_type
        )

    def cancel_order(self, order_id):
        return self.client.cancel_order(order_id=order_id)

    def order_report(self):
        return self.client.order_report()

    def trade_report(self):
        return self.client.trade_report()

    def positions(self):
        return self.client.positions()

    def holdings(self):
        return self.client.holdings()

    def limits(self, segment="ALL", exchange="ALL", product="ALL"):
        return self.client.limits(segment=segment, exchange=exchange, product=product)

    def scrip_master(self, exchange_segment=""):
        return self.client.scrip_master(exchange_segment=exchange_segment)

    def logout(self):
        return self.client.logout()

