import time
import zmq


class TickPublisher:
    """
    ZeroMQ Tick Publisher

    Example
    -------
    pub = TickPublisher(port=5555)

    pub.send({
        "symbol": "NIFTY",
        "ltp": 25120.45,
        "volume": 100
    })
    """

    def __init__(self, host="*", port=5555):
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://{host}:{port}")

    def send(self, tick):
        """Send one tick (dictionary)."""
        self.socket.send_json(tick)

    def stream(self, iterable, delay=0):
        """
        Stream an iterable of dictionaries.

        Parameters
        ----------
        iterable : iterable
            Iterable returning dictionaries.
        delay : float
            Delay between ticks (seconds).
        """
        for tick in iterable:
            self.send(tick)

            if delay > 0:
                time.sleep(delay)

    def close(self):
        self.socket.close()
    
    END_OF_STREAM = "__END__"
    def send_dataframe(self, df, delay=0):
        """
        Send all rows of a DataFrame.

        Each row is converted to a dictionary.
        """
        self.stream(df.to_dict("records"), delay)
        self.send({"type": self.END_OF_STREAM})
        self.close()

class TickSubscriber:
    """
    ZeroMQ Tick Subscriber

    Example
    -------
    sub = TickSubscriber()

    while True:
        tick = sub.recv()
    """

    def __init__(self,
                 host="localhost",
                 port=5555,
                 topic=""):

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.SUB)

        self.socket.connect(f"tcp://{host}:{port}")

        self.socket.setsockopt_string(
            zmq.SUBSCRIBE,
            topic
        )

    def recv(self):
        """Receive one tick."""

        return self.socket.recv_json()

    def listen(self):
        """Infinite generator."""

        while True:
            yield self.recv()

    def close(self):
        self.socket.close()
    
    def recv_df(self):
        """
        Non-blocking receive.

        Returns
        -------
        dict | None
        """
        try:
            return self.socket.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None