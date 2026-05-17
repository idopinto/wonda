void assert(int cond) { if (!(cond)) { ERROR : { reach_error(); abort(); } } }
void assume(int cond) { if (!cond) { abort(); } }
int __VERIFIER_nondet_int();
int main() {
int x = 1;
int m = 1;
int n;
while (x < n) {
if (__VERIFIER_nondet_int()) {
m = x;
}
x = x + 1;
}
if (n > 1) {
assert(m < n);
}
}