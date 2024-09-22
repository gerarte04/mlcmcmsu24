def hello(s: str = '') -> str:
    if len(s) == 0:
        return "Hello!"
    
    return "Hello, " + s + "!" 

def int_to_roman(x: int) -> str:
    ten = {
        1: "I", 10: "X", 100: "C", 1000: "M"
    }
    five = {
        5: "V", 50: "L", 500: "D"
    }

    s = ""
    _del = 1000

    while _del > 0:
        n = x // _del % 10

        if n == 4:
            s += ten[_del] + five[_del * 5]
        elif n == 9:
            s += ten[_del] + ten[_del * 10]
        elif n <= 3:
            for i in range(n):
                s += ten[_del]
        else:
            s += five[_del * 5]
            for i in range(5, n):
                s += ten[_del]

        _del //= 10

    return s

def longest_common_prefix(x: list[str]) -> str:
    if len(x) == 0:
        return ""
    
    s = ""
    st_idx = []

    for i in range(len(x)):
        k = 0

        while k < len(x[i]) and x[i][k].isspace():
            k += 1

        if k == len(x[i]):
            return ""
        
        st_idx.append(k)

    for i in range(len(x[0]) - st_idx[0]):
        for j in range(1, len(x)):
            if st_idx[j] + i >= len(x[j]) or x[j - 1][st_idx[j - 1] + i] != x[j][st_idx[j] + i]:
                return s
        
        s += x[0][st_idx[0] + i]

    return s

class BankCard:
    _total_sum = 0
    _balance_limit = 0

    def __init__(self, total_sum = 0, balance_limit = -1):
        self._total_sum = total_sum
        self._balance_limit = balance_limit

    @property
    def total_sum(self):
        return self._total_sum

    @property
    def balance(self):
        if self._balance_limit == 0:
            raise ValueError("Balance check limits exceeded.")
        
        if self._balance_limit > 0:
            self._balance_limit -= 1
        
        return self._total_sum
    
    @property
    def balance_limit(self):
        return self._balance_limit
    
    def __call__(self, sum_spent):
        if sum_spent > self._total_sum:
            raise ValueError("Not enough money to spend {} dollars.".format(sum_spent))
        
        self._total_sum -= sum_spent
        print("You spent {} dollars.".format(sum_spent))

    def __repr__(self):
        return "To learn the balance call balance."
    
    def __add__(self, another):
        return BankCard(self.total_sum + another.total_sum, max(self.balance_limit, another.balance_limit))

    def put(self, sum_put):
        self._total_sum += sum_put
        print("You put {} dollars.".format(sum_put))

class primes:
    _p = 1

    def __init__(self):
        pass

    @staticmethod
    def is_prime(p):
        d = 2

        while d * d <= p:
            if p % d == 0:
                return False
            d += 1
            
        return True
    
    def __iter__(self):
        return self

    def __next__(self):
        self._p += 1

        while not primes.is_prime(self._p):
            self._p += 1

        return self._p
