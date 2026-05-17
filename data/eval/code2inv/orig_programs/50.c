void assert(int cond) { if (!(cond)) { ERROR : { reach_error(); abort(); } } }
void assume(int cond) { if (!cond) { abort(); } }
int __VERIFIER_nondet_int();
int main() {
int c;
(c = 0);
while (__VERIFIER_nondet_int()) {
{
if (__VERIFIER_nondet_int()) {
if ((c != 4)) {
(c = (c + 1));
}
} else {
if ((c == 4)) {
(c = 1);
}
}
}
}
if ((c != 4)) {
assert((c >= 0));
}
}