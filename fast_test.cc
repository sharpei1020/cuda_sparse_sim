#include <iostream>
#include <cmath>

unsigned round_up_2n(unsigned v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  return v;
}

unsigned dinger_round_up_2n(unsigned v) {
    float f = (v - 1);
    unsigned i = (*(unsigned*)&f) >> 23;
    return 1 << (i - 126);
}

int main() {
    unsigned v = 32;
    std::cout << "v = " << v << std::endl;
    std::cout << "round_up_2n(v) = " << round_up_2n(v) << std::endl;
    std::cout << "dinger_round_up_2n(v) = " << dinger_round_up_2n(v) << std::endl;
    return 0;
}
