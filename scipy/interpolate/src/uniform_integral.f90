module uniform_integral_kernel
  use iso_c_binding, only: c_double, c_int, c_int64_t
  implicit none
contains
  pure real(c_double) function ulp(x)
    real(c_double), intent(in) :: x
    integer(c_int64_t) :: power, bits
    bits = ibclr(transfer(x, 0_c_int64_t), 63)
    power = ibits(bits, 52, 11)
    if (bits == int(z'7fefffffffffffff', c_int64_t)) then
      ulp = transfer(int(z'7ff0000000000000', c_int64_t), 0.0_c_double)
    else if (power <= 52) then
      ulp = transfer(shiftl(1_c_int64_t, int(max(0_c_int64_t, power - 1))), 0.0_c_double)
    else
      ulp = transfer(shiftl(power - 52, 52), 0.0_c_double)
    end if
  end function ulp

  pure real(c_double) function grid_floor(x, origin, step)
    real(c_double), intent(in) :: x, origin, step
    real(c_double) :: q, nearest, product, reconstructed, error
    q = (x - origin) / step
    nearest = anint(q)
    if (abs(q - nearest) == 0.5_c_double) nearest = 2.0_c_double * anint(q * 0.5_c_double)
    product = nearest * step
    reconstructed = origin + product
    error = ulp(x) + ulp(origin) + abs(nearest) * ulp(step) + ulp(product) + ulp(reconstructed)
    if (abs(x - reconstructed) <= error) then
      grid_floor = nearest
    else
      grid_floor = real(floor(q, kind=c_int64_t), c_double)
    end if
  end function grid_floor

  integer(c_int) function uniform_grid_span(t, count, step, first, last) &
      bind(C, name="scipy_uniform_grid_span") result(status)
    integer(c_int64_t), value :: count
    real(c_double), value :: step
    real(c_double), intent(in) :: t(count)
    integer(c_int64_t), intent(out) :: first, last
    integer(c_int64_t) :: i
    real(c_double) :: lower, upper
    status = 1
    lower = huge(step)
    upper = -huge(step)
    do i = 1, count
      if (t(i) /= t(i) .or. abs(t(i)) > huge(step)) return
      lower = min(lower, t(i))
      upper = max(upper, t(i))
    end do
    if (max(abs(lower / step), abs(upper / step)) >= real(huge(first), c_double)) return
    first = int(grid_floor(lower, 0.0_c_double, step), c_int64_t)
    last = int(grid_floor(upper, 0.0_c_double, step), c_int64_t)
    status = 0
  end function uniform_grid_span

  subroutine uniform_integral(t, values, prefixes, anchors, count, cells, tables, first, &
                              step, linear, derivative, output) bind(C, name="scipy_uniform_integral")
    integer(c_int64_t), value :: count, cells, tables, first
    integer(c_int), value :: linear, derivative
    real(c_double), value :: step
    real(c_double), intent(in) :: t(count), values(cells + 1, tables), prefixes(cells, tables), anchors(tables)
    real(c_double), intent(out) :: output(count)
    integer(c_int64_t) :: i, table, cell, block
    real(c_double) :: width, origin, q, fraction, y, delta, integral, block_float
    width = real(cells, c_double) * step
    !$omp parallel do if(count >= 131072) default(none) &
    !$omp shared(t, values, prefixes, anchors, count, cells, tables, first, step, linear, derivative, output, width) &
    !$omp private(i, table, cell, block, origin, q, fraction, y, delta, integral, block_float)
    do i = 1, count
      if (t(i) /= t(i) .or. abs(t(i) / width) >= real(huge(first), c_double) - 1) then
        output(i) = transfer(int(z'7ff8000000000000', c_int64_t), 0.0_c_double)
        cycle
      end if
      block_float = grid_floor(t(i), 0.0_c_double, width)
      block = int(max(real(first, c_double), min(real(first + tables - 1, c_double), block_float)), c_int64_t)
      table = block - first + 1
      origin = real(block, c_double) * width
      q = (t(i) - origin) / step
      cell = int(grid_floor(max(origin, min(origin + width, t(i))), origin, step), c_int64_t)
      cell = max(0_c_int64_t, min(cells - 1, cell))
      fraction = max(0.0_c_double, min(1.0_c_double, q - real(cell, c_double)))
      y = values(cell + 1, table)
      delta = 0.0_c_double
      if (linear /= 0) delta = values(cell + 2, table) - y
      if (derivative /= 0) then
        output(i) = y + fraction * delta
      else
        integral = anchors(table) + prefixes(cell + 1, table)
        if (linear /= 0) then
          output(i) = integral + step * fraction * (y + 0.5_c_double * fraction * delta)
        else
          output(i) = integral + fraction * y * step
        end if
      end if
    end do
    !$omp end parallel do
  end subroutine uniform_integral
end module uniform_integral_kernel
