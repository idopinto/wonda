void assert(int cond) { if (!(cond)) { ERROR : { reach_error(); abort(); } } }
void assume(int cond) { if (!cond) { abort(); } }
int __VERIFIER_nondet_int();
int main() {
int i = 0;
int j, c, t;
while (__VERIFIER_nondet_int()) {
if (c > 48) {
if (c < 57) {
j = i + i;
t = c - 48;
i = j + t;
}
}
}
assert(i >= 0);
}